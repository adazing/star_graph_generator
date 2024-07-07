import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
# from hellaswag import render_example, iterate_examples
from torch.utils.tensorboard import SummaryWriter

"""
We need to define the following models:
z[t] = enc_F(x[1:t])                    | Forward encoder
z'[t+k] = enc_B(x[t+k:T, ::-1])         | Backward encoder
e'[t] = FSQ(z[t])                       | Finite State Quantization
e'[t+k/2] = B(z[t], e'[t])              | Backward Goal Predictor
"""

# Architecture -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class ForwardEncoderConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    reverse_input: bool = False 

@dataclass
class BackwardEncoderConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    reverse_input: bool = True

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.reverse_input = config.reverse_input
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        if self.reverse_input:
            idx = idx.flip(1)
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm
        x = self.transformer.ln_f(x)
        return x

@dataclass
class DynamicsConfig:
    vocab_size: int # Size of the classification label space.
    n_goal: int # Size of the goal embedding
    block_size: int = 1024 # max sequence length
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_f_enc: int = 768 # embedding dimension
    n_embd: int = 768 # embedding dimension

# TODO: (Edward) try a more sophisticated model for the dynamics, like transformer or do FILM between z_t and e'_t+k.
class Dynamics(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_f_enc + config.n_goal, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, forward_embedding, backward_goal, targets=None):
        x = torch.cat([forward_embedding, backward_goal], dim=-1)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

@dataclass
class TextHeadConfig:
    n_goal: int # Size of the goal embedding
    vocab_size: int = 50304 # number of tokens
    block_size: int = 1024 # max sequence length
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_f_enc: int = 768 # forward latent dimension
    n_embd: int = 768 # embedding dimension

class TextHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer norm
        self.ln_f = nn.LayerNorm(config.n_embd + config.n_goal)
        self.lm_head = nn.Linear(config.n_f_enc + config.n_goal, config.vocab_size, bias=False)
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, forward_embedding, backward_embedding, targets=None):
        x = torch.concat([forward_embedding, backward_embedding], dim=-1)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss 

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "tinystories2-ascii"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # x = (buf[:-1]).view(B, T) # inputs
        # y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return buf

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

from fsq import FSQ
# create model
f_enc = Encoder(ForwardEncoderConfig(n_layer=6, block_size=1024))
b_enc = Encoder(BackwardEncoderConfig(n_layer=6, block_size=1024))
# qtz = FSQ(levels=[8,8,8], dim=b_enc.config.n_embd) # 512 codebook size
# print('Codebook size:', qtz.codebook_size)
# b_dyn = Dynamics(DynamicsConfig(vocab_size=qtz.codebook_size, n_goal=b_enc.config.n_embd))
text_head = TextHead(TextHeadConfig(n_goal=b_enc.config.n_embd))

models = [f_enc, b_enc, 
        #   qtz, 
        #   b_dyn, 
          text_head]
for m in models:
    m.to(device)

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    f_enc = torch.compile(f_enc)
    b_enc = torch.compile(b_enc)
    # qtz = torch.compile(qtz)
    # b_dyn = torch.compile(b_dyn)
    text_head = torch.compile(text_head)
if ddp:
    f_enc = DDP(f_enc, device_ids=[ddp_local_rank])
    b_enc = DDP(b_enc, device_ids=[ddp_local_rank])
    # qtz = DDP(qtz, device_ids=[ddp_local_rank])
    # b_dyn = DDP(b_dyn, device_ids=[ddp_local_rank])
    text_head = DDP(text_head, device_ids=[ddp_local_rank])

models = dict(f_enc=f_enc, b_enc=b_enc, 
            #   qtz=qtz, 
            #   b_dyn=b_dyn, 
              text_head=text_head)
raw_models = dict(
    raw_f_enc = f_enc.module if ddp else f_enc,
    raw_b_enc = b_enc.module if ddp else b_enc,
    # raw_qtz = qtz.module if ddp else qtz,
    # raw_b_dyn = b_dyn.module if ddp else b_dyn,
    raw_text_head = text_head.module if ddp else text_head,
)

# TODO: (Edward) this LR schedule is specific to fineweb data size
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
def configure_optimizers(models, weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for m in models for pn, p in m.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer
optimizer = configure_optimizers(raw_models.values(), weight_decay=0.0, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
writer = SummaryWriter(log_dir)


text_loss_weight = 1
def compute_losses(x):
    """
    token prediction problem: given a sequence X, predict X[t+1] given the starting state forward(X[1:t]) and goal state backward(X[t+2:T]) .To implement this, we will use enc_f as the forward and enc_b as the backward encoder 

    let's say the sequence is [1,2,3,4,5]. we will construct the input and output pairs as follows:
    If we feed the sequence to the forward encoder, we get the following hidden states:
    [f(1), f(12), f(123), f(1234), f(12345)] 
    remember that the backward transformer reverses the sequence. So if we feed it [1,2,3,4,5],
    [b(5), b(54), b(543), b(5432), b(54321)] for the backward transformer hiddden states

    Now, we can construct the input and output pairs:
    x = [ (f(1), b(543)), (f(12), b(54)), (f(123), b(5)) ]
    y = [ 2, 3, 4]
    """
    forward = f_enc(x) # [f(1), f(12), f(123), f(1234), f(12345)]
    backward = b_enc(x) # [b(5), b(54), b(543), b(5432), b(54321)]
    """
    def make_prod(S,SB,A, mink=0, maxk=15, pivot=0):
    #A = A.squeeze()
    row_indices_S = torch.arange(S.shape[0]).cuda()
    row_indices_SB = torch.arange(SB.shape[0]).cuda()
    combinations = torch.cartesian_prod(row_indices_S, row_indices_SB)
    st = S[combinations[:, 0]]
    stk = SB[combinations[:, 1]]
    a = A[combinations[:,0]]
 
    k = combinations[:,1]-combinations[:,0] + pivot
 
    st = st[(k > mink) & (k <= maxk)]
    stk = stk[(k > mink) & (k <= maxk)]
    a = a[(k > mink) & (k <= maxk)]
    k = k[(k > mink) & (k <= maxk)]
 
    return st, stk, k, a
    """
    import ipdb; ipdb.set_trace()



    flip_backward = backward.flip(1) # [b(54321), b(5432), b(543), b(54), b(5)]

    text_f = forward[:, :-2] # [f(1), f(12), f(123)]
    text_b = flip_backward[:, 2:] # [b(543), b(54), b(5)]
    # codes, indices = qtz(text_b)
    indices = None
    text_labels = x[:, 1:-1] # [ 2, 3, 4]
    text_logits, text_loss = text_head(text_f, text_b, targets=text_labels)
    
    """
    construct input and output pairs for a given K from the sequence
    where x is the tuple of the start and end element, and y is the midpoint 
    So for k=1, we want x to be [(f(1), b(543)), (f(12), b(54)), (f(123),b(5))] and y to be [b(5432), b(543), b(54)]
    And if k=2, we want x to be [ (f(1), b(5)) ] and y to be [b(543)]
    """
    # TODO: Use cartesian product to generate the pairs
    k_step_losses = {1:torch.zeros((1,))}
    # for k in [1,2,4,8, 16, 32, 64, 128, 256]:
    #     all_starts = forward[:, :-2 * k]                  # [ f(1), f(12), f(123)]
    #     all_ends, _ = qtz(flip_backward[:, 2 * k:])       # [ b(543), b(54), b(5)]
    #     _, all_mids_indices = qtz(flip_backward[:, k:-k]) # [ b(5432), b(543), b(54)]
    #     _, k_loss = b_dyn(all_starts, all_ends, targets=all_mids_indices)
    #     k_step_losses[k] = k_loss
    return text_loss, k_step_losses, indices



sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42 + ddp_rank)

def bplan(forward, subgoal, k):
    # print(f"bplan: k={k}, forward={forward.shape}, subgoal={subgoal.shape}")
    if k <= 1:
        text_logits, _ = text_head(forward, subgoal)
        # take the logits at the last position
        text_logits = text_logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(text_logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        return xcol
    # import ipdb; ipdb.set_trace()
    subgoal_logits, _ = b_dyn(forward, subgoal)
    subgoal_logits = subgoal_logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(subgoal_logits, dim=-1)
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    # note: multinomial does not demand the input to sum to 1
    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
    # gather the corresponding indices
    chosen_ix = torch.gather(topk_indices, -1, ix) # (B, 1)
    codes = qtz.indices_to_codes(chosen_ix)
    return bplan(forward, codes, k // 2)


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 10 == 0 or last_step:
        for m in models.values():
            m.eval()

        val_loader.reset()
        with torch.no_grad():
            val_k_step_losses_accum = {}
            val_text_loss_accum = 0.0
            val_loss_accum = 0.0
            val_loss_steps = 20
            used_codes = set()
            for _ in range(val_loss_steps):
                buf = val_loader.next_batch()
                buf = buf.to(device)
                x = buf[:-1].view(B, T)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    text_loss, k_step_losses, code_indices = compute_losses(x)
                    # code_indices = code_indices.view(-1)
                    # for quan_idx in code_indices:
                    #     used_codes.add(quan_idx.item())
                
                loss = text_loss_weight * text_loss + torch.sum(torch.stack([v for v in k_step_losses.values()]))
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                val_text_loss_accum += text_loss.detach() / val_loss_steps # need to divide here since we didn't divide text_loss
                for k, k_loss in k_step_losses.items():
                    if k not in val_k_step_losses_accum:
                       val_k_step_losses_accum[k] = 0.0
                    val_k_step_losses_accum[k] += k_loss.detach() / val_loss_steps
            # val_codebook_usage = len(used_codes) / qtz.codebook_size
            used_codes.clear()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_text_loss_accum, op=dist.ReduceOp.AVG)
            for v in val_k_step_losses_accum.values():
                dist.all_reduce(v, op=dist.ReduceOp.AVG)
            # dist.all_reduce(val_codebook_usage, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            writer.add_scalar("loss/val", val_loss_accum.item(), step)
            writer.add_scalar("text_loss/val", val_text_loss_accum.item(), step)
            for k,v in val_k_step_losses_accum.items():
                writer.add_scalar(f"{k}_step_loss/val", v.item(), step)
            # writer.add_scalar("codebook_usage/val", val_codebook_usage, step)
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                for name, model in raw_models.items():
                    checkpoint[name] = model.state_dict()
                    if hasattr(model, 'config'):
                        checkpoint[name+'_config'] = model.config
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

        # do one step of the optimization
    
    # once in a while generate from the model (except step 0, which is noise)
    # if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    #     for m in models.values():
    #         m.eval()
    #     num_return_sequences = 4
    #     max_length = 32
    #     tokens = enc.encode("Once upon a time,")
    #     tokens = torch.tensor(tokens, dtype=torch.long)
    #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     xgen = tokens.to(device)

    #     goal_tokens = enc.encode("happily ever after.")
    #     goal_tokens = torch.tensor(goal_tokens, dtype=torch.long)
    #     goal_tokens = goal_tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     goal_tokens = goal_tokens.to(device)
        
    #     while xgen.size(1) < max_length:
    #         # forward the model to get the logits
    #         with torch.no_grad():
    #             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    #                 forward = f_enc(xgen)[:, -1:, :]
    #                 backward = b_enc(goal_tokens)[:, -1:, :] 
    #                 codes, indices = qtz(backward)
    #                 k = max_length - xgen.size(1)
    #                 # print(f"Calling bplan with k={k}")
    #                 xcol = bplan(forward, codes, k)
    #                 xgen = torch.cat((xgen, xcol), dim=1)

    #     # print the generated text
    #     all_decoded = ""
    #     for i in range(num_return_sequences):
    #         tokens = xgen[i, :max_length].tolist()
    #         decoded = enc.decode(tokens)
    #         all_decoded += f"sample {i}: {decoded}\n"
    #         print(f"rank {ddp_rank} sample {i}: {decoded}")
    #     if master_process:
    #         writer.add_text(f"sampled_text", all_decoded, step)


    # do one step of the optimization
    for m in models.values():
        m.train()
    optimizer.zero_grad()
    k_step_losses_accum = {}
    text_loss_accum = 0.0
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        buf = train_loader.next_batch()
        buf = buf.to(device)
        x = buf[:-1].view(B, T)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            text_loss, k_step_losses, _ = compute_losses(x)
        loss = text_loss_weight * text_loss + torch.sum(torch.stack([v for v in k_step_losses.values()]))
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps

        # for logging losses
        loss_accum += loss.detach()
        text_loss_accum += text_loss.detach() / grad_accum_steps # need to divide here since we didn't divide text_loss
        for k, k_loss in k_step_losses.items():
            if k not in k_step_losses_accum:
                k_step_losses_accum[k] = 0.0
            k_step_losses_accum[k] += k_loss.detach() / grad_accum_steps

        if ddp:
            for m in models.values():
                m.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(text_loss_accum, op=dist.ReduceOp.AVG)
        for v in k_step_losses_accum.values():
            dist.all_reduce(v, op=dist.ReduceOp.AVG)

    norm_stats = {}
    for name, model in models.items():
        norm_stats[name+'_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    total_norm = sum(norm_stats.values())

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        writer.add_scalar("loss/train", loss_accum.item(), step)
        writer.add_scalar("text_loss/train", text_loss_accum.item(), step)
        for k,v in k_step_losses_accum.items():
            writer.add_scalar(f"{k}_step_loss/train", v.item(), step)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("norm/total", total_norm, step)
        for name, norm in norm_stats.items():
            writer.add_scalar(f"norm/{name}", norm, step)
        writer.add_scalar("dt", dt, step)
        writer.add_scalar("tps", tokens_per_sec, step)

        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {total_norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()