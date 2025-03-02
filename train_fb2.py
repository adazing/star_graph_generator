import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
# from hellaswag import render_example, iterate_examples
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import tiktoken
import config
import numpy as np
from evaluate import evaluate2
"""
We need to define the following models:
z[t] = enc_F(x[1:t])                    | Forward encoder
z'[t+k] = enc_B(x[t+k:T, ::-1])         | Backward encoder
x[t+1] = text_head(z[t], z'[t+k])       | Text head

The models are trained on the cartesian product of the forward and backward encodings. Since the cartesian product is too large to fit in memory, we subsample it. The forward and backward encodings are computed for the entire sequence, and then indexed into for the minibatches.
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
    n_layer: int = 3 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 384 # embedding dimension
    reverse_input: bool = False 

@dataclass
class BackwardEncoderConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens
    n_layer: int = 3 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 384 # embedding dimension
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
class TextHeadConfig:
    n_goal: int # Size of the goal embedding
    vocab_size: int = 50304 # number of tokens
    block_size: int = 1024 # max sequence length
    n_layer: int = 3 # number of layers
    n_head: int = 6 # number of heads
    n_f_enc: int = 384 # forward latent dimension
    n_embd: int = 384 # embedding dimension

class TextHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer norm
        # self.ln_f = nn.LayerNorm(config.n_embd + config.n_goal)
        # self.lm_head = nn.Linear(config.n_f_enc + config.n_goal, config.vocab_size, bias=False)
        # Linear, LeakyRelu, Linear
        self.lm_head = nn.Sequential(
            nn.Linear(config.n_f_enc + config.n_goal, 512, bias=False),
            nn.LeakyReLU(),
            nn.Linear(512, config.vocab_size, bias=False),
        )
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
    
    def forward(self, forward_embedding, backward_embedding, targets=None, return_info=False):
        x = torch.concat([forward_embedding, backward_embedding], dim=-1)
        # x = self.ln_f(x)
        # logits = self.lm_head(x)
        logits = self.lm_head(x)
        loss = None 
        info = {}
        if targets is not None:
            if return_info:
                _loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
                info['loss_before_mean'] = _loss
                loss = _loss.mean()
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        if return_info:
            return logits, loss, info
        else:
            return logits, loss

    def generate(self, f_enc, b_enc, idx, goal, max_length, temperature=1.0, top_k = 1):
        # f_enc = forward_embedding(x)
        # b_enc = backward_embedding(x)
        f_enc.eval()
        b_enc.eval()
        self.eval()
        device = "cuda"
        f_enc.to(device)
        b_enc.to(device)
        generated = idx  # Start with the input indices
        generated = generated.to(device)
        # device = idx.device
        # graph_description_length = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3
        # fb_pairs, labels, dt, midpoints = create_cartesian_product_examples(T-graph_description_length)
        # fb_pairs += graph_description_length
        # labels += graph_description_length
        # midpoints += graph_description_length

        # subset_size = len(fb_pairs)
        # minibatch_size = 2**13 + 2**11 + 2**10
        # subsample_ratio = subset_size / len(fb_pairs)
        
        # num_minibatches = -(-subset_size // minibatch_size)

        # with torch.no_grad():
        #     forward = f_enc(x) # [f(1), f(12), f(123), f(1234), f(12345)]
        #     backward = b_enc(x) # [b(5), b(54), b(543), b(5432), b(54321)]
        #     _backward = backward.flip(1) # [b(54321), b(5432), b(543), b(54), b(5)]

        # subsampled_idxs = torch.randint(high=len(fb_pairs), size=(subset_size,))
        # _fb_pairs = fb_pairs[subsampled_idxs]
        # _labels = labels[subsampled_idxs]
        # _dt = dt[subsampled_idxs]
        # _midpoints = midpoints[subsampled_idxs]
        

        while generated.size(1) < max_length:
            # Get the last token in the current sequence to use as input
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    forward = f_enc(generated)[:, -1:, :] # gets f(123)
                    # print(forward.shape)
                    backward = b_enc(goal)[:, -1:, :] # gets b(987) from [b(9), b(98), b(987)]
                    # print(backward.shape)
                    # logits, _ = text_head(forward, backward)
                    # Generate logits for the current sequence
                    logits, _ = self.forward(forward, backward)
                    # Take the logits from the last time step
                    logits = logits[:, -1, :] / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('inf')
                
                    # Convert logits to probabilities
                    probs = F.softmax(logits, dim=-1)
                    # Sample from the probabilities to get the next token
                    next_token = torch.multinomial(probs, num_samples=1)
                    # Append the new token to the sequence
                    # print(generated.shape)
                    # print(next_token.shape)
                    generated = torch.cat((generated, next_token), dim=1)

                    # _f = forward[:, _fb_pairs[i:i+minibatch_size, 0]]
                    # _b = _backward[:, _fb_pairs[i:i+minibatch_size, 1]]
                    # text_labels = x[:, _labels[i:i+minibatch_size]]
                    # logits, loss, loss_info = text_head(_f, _b, targets=text_labels, return_info=True)


        f_enc.train()
        b_enc.train()
        self.train()
        return generated

#################################################################
            # x = train_loader.next_batch()
            # x = x.to(device)
            # # TODO: look into why EOS tokens (id 1) are missing
            # # tokens = tokenizer(text, return_tensors="pt", max_length=T, padding='max_length', truncation=True)['input_ids']
            # # x = torch.as_tensor(tokens, device=device_type)
            # graph_description_length = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3
            # fb_pairs, labels, dt, midpoints = create_cartesian_product_examples(T-graph_description_length)
            # # print(fb_pairs, labels, dt, midpoints)
            # fb_pairs += graph_description_length
            # labels += graph_description_length
            # midpoints += graph_description_length
            # # usually set this to some number lower than len(fb_pairs)
            # subset_size = len(fb_pairs)
            # minibatch_size = 2**13 + 2**11 + 2**10
            # subsample_ratio = subset_size / len(fb_pairs)
            # # print("subset size", subset_size, "subsample ratio", subsample_ratio)
            # num_minibatches = -(-subset_size // minibatch_size)
            # # print("minibatch size", minibatch_size, "num_minibatches:", num_minibatches)
            
            # forward = f_enc(x) # [f(1), f(12), f(123), f(1234), f(12345)]
            # backward = b_enc(x) # [b(5), b(54), b(543), b(5432), b(54321)]
            # _backward = backward.flip(1) # [b(54321), b(5432), b(543), b(54), b(5)]

            # # subsample the pairs, since we can't fit all of them in memory
            # subsampled_idxs = torch.randint(high=len(fb_pairs), size=(subset_size,))
            # _fb_pairs = fb_pairs[subsampled_idxs]
            # _labels = labels[subsampled_idxs]
            # _dt = dt[subsampled_idxs]
            # _midpoints = midpoints[subsampled_idxs]

            # for i in range(0, len(_fb_pairs), minibatch_size):
            #     with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            #         _f = forward[:, _fb_pairs[i:i+minibatch_size, 0]]
            #         _b = _backward[:, _fb_pairs[i:i+minibatch_size, 1]]
            #         text_labels = x[:, _labels[i:i+minibatch_size]]
            #         logits, loss, loss_info = text_head(_f, _b, targets=text_labels, return_info=True)
            #         loss_before_mean = loss_info['loss_before_mean']
            #         _dt = _dt[None].repeat(B, 1)
            #         # print(logits, loss, loss_info)
            #         # print(_dt.shape)
            #         # print(_dt)
            #         print("k = 2 loss:", loss_before_mean[_dt.view(-1)==2].mean().item())
            #         print("k = rand loss:", loss_before_mean[_dt.view(-1)!=2].mean().item())
                    
            #         # conditional VAE stuff here.
            #         # import ipdb; ipdb.set_trace()
            #         def loss_latent_inpaint(starts, goals, targets, deltas):
            #             # get _b_mid
            #             # get k and embedding layer.
            #             pass

            #         lip_loss = loss_latent_inpaint(_f, _b, _backward[:, midpoints], _dt)
                    
                    
            #         # decode some examples to text space for visualization.
            #         if False:
            #         # if step % 100 == 0 and i == 0:
            #             print(f"Step {step}: Train inputs, predictions, and labels")
            #             for i in range(4):
            #                 vforward_tokens = tokens[0, :_fb_pairs[i, 0] + 1]
            #                 vbackward_tokens = tokens[0, _fb_pairs[i, 1]:]
            #                 vlabels = x[0, _labels[i]]
            #                 vlogits = logits[0, i]

            #                 # top k sampling
            #                 # get the probabilities
            #                 probs = F.softmax(vlogits, dim=-1)
            #                 # do top-k sampling of 50 (huggingface pipeline default)
            #                 # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            #                 topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            #                 # select a token from the top-k probabilities
            #                 # note: multinomial does not demand the input to sum to 1
            #                 ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            #                 # gather the corresponding indices
            #                 xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            #                 # append to the sequence
                            
                            
            #                 print("FORWARD:", tokenizer.decode(vforward_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=True))
            #                 print("\n")
            #                 print("label:", tokenizer.decode(vlabels))
            #                 print("pred:", tokenizer.decode(xcol))
            #                 print("\n")
            #                 print("BACKWARD:", tokenizer.decode(vbackward_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                           
            #                 print("-" * 80)
                            

# Original cartesian product functoin
# def create_cartesian_product_examples(T):
#     # precompute timestep pairs (t, t+k) for the forward and backward encoders
#     ft = torch.arange(T, dtype=torch.int32).cuda() # [0, 1 .... T-1]
#     bt = torch.arange(T, dtype=torch.int32).cuda() # [0, 1 .... T-1]
#     combinations = torch.cartesian_prod(ft, bt) # [(i,j) for i in ft for j in bt]
#     dt = combinations[:, 1] - combinations[:, 0] # [(j - i)] n^2 of these deltas
#     max_dt = 2 # corresponds to K=1.

#     # we want all pairs where dt is greater than 2 and less than max_dt
#     fb_pairs = combinations[(dt >=  2) & (dt <= max_dt)]
#     labels = (combinations[:, 0] + 1)[(dt >=  2) & (dt <= max_dt)]
#     return fb_pairs, labels

# modified K=2 cartesian product function
# def create_cartesian_product_examples(T):
#     # precompute timestep pairs (t, t+k) for the forward and backward encoders
#     ft = torch.arange(T-2, dtype=torch.int32).cuda() # [0, 1 .... T-2]
#     dt = torch.Tensor([2]).type(torch.int32).to(device_type)
#     combinations = torch.cartesian_prod(ft, dt) 
#     combinations[:, 1] = combinations[:, 0] + combinations[:, 1]
#     # bt = combinations[:, 0] + combinations[:, 1] 
#     fb_pairs = combinations
#     labels = (combinations[:, 0] + 1)
#     return fb_pairs, labels

# modified K=2, K=rand cartesian product function
import random
def create_cartesian_product_examples(T):
    # precompute timestep pairs (t, t+k) for the forward and backward encoders
    ft = torch.arange(T, dtype=torch.int32).to(device_type) # [0, 1 .... T-1]
    # k_rand = random.randint(3, T-2) # or sample from a logarithimic
    # dt = torch.Tensor([2, k_rand]).type(torch.int32).to(device_type)
    k_rand = 2
    dt = torch.Tensor([k_rand]).type(torch.int32).to(device_type)
    combinations = torch.cartesian_prod(ft, dt)

    combinations[:, 1] = combinations[:, 0] + combinations[:, 1]
    midpoints = combinations[:, 0] + (combinations[:, 1] - combinations[:, 0]) // 2
    
    # bt = combinations[:, 0] + combinations[:, 1]
    fb_pairs = combinations.clone()
 
    #make sure no fb_pair goes over limit.
 
    labels = (combinations[:, 0] + 1)
 
    fb_pairs = fb_pairs[combinations[:,1] < T]
    labels = labels[combinations[:,1] < T]
    midpoints = midpoints[combinations[:,1] < T]
    
    # (f_t, b_{t+k+1}), (x_{t+k}), (dt), (b_{t+k//2})
    return fb_pairs, labels, fb_pairs[:,1] - fb_pairs[:, 0], midpoints

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
        data_root = "tokenized_data"
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
        # print(self.tokens.shape)
        buf = self.tokens[self.current_position : self.current_position+B*T]
        x = buf.clone().view(B, T)
        # x = buf.clone().view(B, T)[:, :-1] # inputs
        # y = buf.clone().view(B, T)[:, 1:] # targets
        # y[:, :-config.lenOfEachPath] = -1 # empty
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
if __name__ == "__main__":
    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    results = {}

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

    B = 64 # micro batch size
    T = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3 + config.lenOfEachPath # sequence length
    total_batch_size = B * T # 2**19, ~0.5M, in number of tokens

    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    torch.set_float32_matmul_precision('high')

    from fsq import FSQ
    # from stochastic_predictor import StochasticPredictor
    # create model
    f_enc = Encoder(ForwardEncoderConfig(n_layer=3, block_size=T, n_head=6, vocab_size=config.maxNodes + 4))
    b_enc = Encoder(BackwardEncoderConfig(n_layer=3, block_size=T, n_head=6, vocab_size=config.maxNodes + 4))
    text_head = TextHead(TextHeadConfig(n_goal=b_enc.config.n_embd, vocab_size=config.maxNodes + 4))

    # sp = StochasticPredictor(768 * 2 + 256, 768).cuda()

    models = [f_enc, b_enc, text_head]
    for m in models:
        m.to(device)

    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        f_enc = torch.compile(f_enc)
        b_enc = torch.compile(b_enc)
        text_head = torch.compile(text_head)
    if ddp:
        f_enc = DDP(f_enc, device_ids=[ddp_local_rank])
        b_enc = DDP(b_enc, device_ids=[ddp_local_rank])
        text_head = DDP(text_head, device_ids=[ddp_local_rank])

    models = dict(f_enc=f_enc, b_enc=b_enc, text_head=text_head)
    raw_models = dict(
        raw_f_enc = f_enc.module if ddp else f_enc,
        raw_b_enc = b_enc.module if ddp else b_enc,
        raw_text_head = text_head.module if ddp else text_head,
    )

    # Learning hyperparameters from the video.
    # max_lr = 6e-4
    # min_lr = max_lr * 0.1
    # warmup_steps = 715
    # max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    # def get_lr(it):
    #     # 1) linear warmup for warmup_iters steps
    #     if it < warmup_steps:
    #         return max_lr * (it+1) / warmup_steps
    #     # 2) if it > lr_decay_iters, return min learning rate
    #     if it > max_steps:
    #         return min_lr
    #     # 3) in between, use cosine decay down to min learning rate
    #     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    #     assert 0 <= decay_ratio <= 1
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    #     return min_lr + coeff * (max_lr - min_lr)

    # Learning hyperparameters for tinystories
    epochs = 100
                #         number of batches total  =  validation data token size /  batch_size * length of each line
    max_steps = epochs * (config.numOfSamples//B - (config.shard_size // (B * (config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3 + config.lenOfEachPath)))) # 1000 steps is ~1 epoch, if data is 500M tokens and batch size is 0.5M tokens
    # def get_lr(it):
    #     return 3e-4

    eval_every = 5000
    
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
    optimizer = configure_optimizers(raw_models.values(), weight_decay=0.0, learning_rate=1e-4, device_type=device_type)

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    # use the current date and time as a unique identifier for this training run
    import datetime
    now = datetime.datetime.now()
    log_dir = os.path.join(log_dir, now.strftime("%Y-%m-%d-%H-%M-%S"))

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    writer = SummaryWriter(log_dir)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    # tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", eos_token="<|eos|>", pad_token="<|pad|>")
    # dataset, eval_dataset = load_dataset('tokenized_data', split=['train', 'validation'], download_mode="reuse_cache_if_exists")
    # dataset.set_format(type='torch', columns=['text'], device=device_type)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64) 

    # eval_dataset.set_format(type='torch', columns=['text'], device=device_type)
    # eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=64) 

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    eval_dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # infinite data loader
    # def cycle(loader):
    #     while True:
    #         # check if iterable has exahusted
    #         yield loader.next_batch()

    # inf_dataloader = cycle(train_loader)



    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        # if False:
        # # if step % 250 == 0 or last_step:
        #     for m in models.values():
        #         m.eval()
        #     with torch.no_grad():
        #         val_loss_accum = 0.0
        #         val_loss_steps = 20
        #         used_codes = set()
        #         pbar = tqdm(range(val_loss_steps), desc=f"Calculating validation loss.", disable=False)
        #         for val_step, data in enumerate(eval_dataloader):
        #             text = data['text']
        #             tokens = tokenizer(text, return_tensors="pt", max_length=1024, padding=True, truncation=True)['input_ids']
        #             x = torch.as_tensor(tokens, device=device_type)
        #             T = x.size(1)
        #             fb_pairs, labels = create_cartesian_product_examples(T)
        #             # usually set this to some number lower than len(fb_pairs)
        #             subset_size = len(fb_pairs)
        #             minibatch_size = 2**16 + 2**14
        #             subsample_ratio = subset_size / len(fb_pairs)
        #             # print("subset size", subset_size, "subsample ratio", subsample_ratio)
        #             num_minibatches = -(-subset_size // minibatch_size)
        #             # print("minibatch size", minibatch_size, "num_minibatches:", num_minibatches)

        #             forward = f_enc(x) # [f(1), f(12), f(123), f(1234), f(12345)]
        #             backward = b_enc(x) # [b(5), b(54), b(543), b(5432), b(54321)]
        #             _backward = backward.flip(1) # [b(54321), b(5432), b(543), b(54), b(5)]

        #             for i in range(0, len(fb_pairs), minibatch_size):
        #                 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #                     _f = forward[:, fb_pairs[i:i+minibatch_size, 0]]
        #                     _b = _backward[:, fb_pairs[i:i+minibatch_size, 1]]
        #                     text_labels = x[:, labels[i:i+ minibatch_size]]
        #                     loss = text_head(_f, _b, targets=text_labels)[1]
                        
        #                 loss = loss / (val_loss_steps * num_minibatches)
        #                 val_loss_accum += loss.detach()
        #             pbar.update(1)
        #             if val_step > val_loss_steps:
        #                 break
        #         pbar.close()
        #     if ddp:
        #         dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        #     if master_process:
        #         print(f"validation loss: {val_loss_accum.item():.4f}")
        #         with open(log_file, "a") as f:
        #             f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        #         writer.add_scalar("loss/val", val_loss_accum.item(), step)
        #         if step > 0 and (step % 250 == 0 or last_step):
        #             # optionally write model checkpoints
        #             checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        #             checkpoint = {
        #                 'step': step,
        #                 'val_loss': val_loss_accum.item()
        #             }
        #             for name, model in raw_models.items():
        #                 checkpoint[name] = model.state_dict()
        #                 if hasattr(model, 'config'):
        #                     checkpoint[name+'_config'] = model.config
        #             # you might also want to add optimizer.state_dict() and
        #             # rng seeds etc., if you wanted to more exactly resume training
        #             torch.save(checkpoint, checkpoint_path)


        # do one step of the optimization
        for m in models.values():
            m.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        # pbar = tqdm(range(grad_accum_steps * num_minibatches), desc=f"Performing grad accumulation.", disable=False)
        grad_accum_steps = 1
        for micro_step in range(grad_accum_steps):
            # print("micro_step", micro_step)
            # text = next(inf_dataloader)['text']
            x = train_loader.next_batch()
            x = x.to(device)
            # print(x)
            # TODO: look into why EOS tokens (id 1) are missing
            # tokens = tokenizer(text, return_tensors="pt", max_length=T, padding='max_length', truncation=True)['input_ids']
            # x = torch.as_tensor(tokens, device=device_type)
            graph_description_length = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3
            fb_pairs, labels, dt, midpoints = create_cartesian_product_examples(T-graph_description_length)
            # print(fb_pairs, labels, dt, midpoints)
            fb_pairs += graph_description_length
            labels += graph_description_length
            midpoints += graph_description_length
            # usually set this to some number lower than len(fb_pairs)
            subset_size = len(fb_pairs)
            # minibatch_size = 2**13 + 2**11 + 2**10
            minibatch_size = len(fb_pairs)  # TODO
            subsample_ratio = subset_size / len(fb_pairs)
            # print("subset size", subset_size, "subsample ratio", subsample_ratio)
            num_minibatches = -(-subset_size // minibatch_size)
            # print("minibatch size", minibatch_size, "num_minibatches:", num_minibatches)
            
            forward = f_enc(x) # [f(1), f(12), f(123), f(1234), f(12345)]
            backward = b_enc(x) # [b(5), b(54), b(543), b(5432), b(54321)]
            _backward = backward.flip(1) # [b(54321), b(5432), b(543), b(54), b(5)]

            # subsample the pairs, since we can't fit all of them in memory
            # subsampled_idxs = torch.randint(high=len(fb_pairs), size=(subset_size,))  # TODO, number maybe repeated
            # _fb_pairs = fb_pairs[subsampled_idxs]
            # _labels = labels[subsampled_idxs]
            # _dt = dt[subsampled_idxs]
            # _midpoints = midpoints[subsampled_idxs]
            _fb_pairs = fb_pairs
            _labels = labels
            _dt = dt
            _midpoints = midpoints

            # for i in range(0, len(_fb_pairs), minibatch_size):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # _f = forward[:, _fb_pairs[i:i+minibatch_size, 0]]
                # _b = _backward[:, _fb_pairs[i:i+minibatch_size, 1]]
                # text_labels = x[:, _labels[i:i+minibatch_size]]
                _f = forward[:, _fb_pairs[:, 0]]
                _b = _backward[:, _fb_pairs[:, 1]]
                text_labels = x[:, _labels]
                logits, loss, loss_info = text_head(_f, _b, targets=text_labels, return_info=True)
                loss_before_mean = loss_info['loss_before_mean']
                # _dt = _dt[None].repeat(B, 1)
                # print(logits, loss, loss_info)
                # print(_dt.shape)
                # print(_dt)
                # print(loss_before_mean)
                # print(_dt)
                # print(_fb_pairs)
                # print("k = 2 loss:", loss_before_mean.mean().item())

                # print("k = 2 loss:", loss_before_mean[_dt.view(-1)==2].mean().item())
                # print("k = rand loss:", loss_before_mean[_dt.view(-1)!=2].mean().item())

                # conditional VAE stuff here.
                # import ipdb; ipdb.set_trace()
                def loss_latent_inpaint(starts, goals, targets, deltas):
                    # get _b_mid
                    # get k and embedding layer.
                    pass

                # lip_loss = loss_latent_inpaint(_f, _b, _backward[:, midpoints], _dt)


                # decode some examples to text space for visualization.
                # if False:
                # # if step % 100 == 0 and i == 0:
                #     print(f"Step {step}: Train inputs, predictions, and labels")
                #     for i in range(4):
                #         vforward_tokens = tokens[0, :_fb_pairs[i, 0] + 1]
                #         vbackward_tokens = tokens[0, _fb_pairs[i, 1]:]
                #         vlabels = x[0, _labels[i]]
                #         vlogits = logits[0, i]
                #
                #         # top k sampling
                #         # get the probabilities
                #         probs = F.softmax(vlogits, dim=-1)
                #         # do top-k sampling of 50 (huggingface pipeline default)
                #         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                #         # select a token from the top-k probabilities
                #         # note: multinomial does not demand the input to sum to 1
                #         ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                #         # gather the corresponding indices
                #         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                #         # append to the sequence
                #
                #
                #         print("FORWARD:", tokenizer.decode(vforward_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=True))
                #         print("\n")
                #         print("label:", tokenizer.decode(vlabels))
                #         print("pred:", tokenizer.decode(xcol))
                #         print("\n")
                #         print("BACKWARD:", tokenizer.decode(vbackward_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #
                #         print("-" * 80)

            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / (grad_accum_steps * num_minibatches)

            # for logging losses
            loss_accum += loss.detach()
            # last_step = (micro_step == grad_accum_steps - 1) and (i + minibatch_size >= len(fb_pairs))
            if ddp:
                for m in models.values():
                    m.require_backward_grad_sync = last_step
            # loss.backward(retain_graph=not last_step)
            loss.backward()
            # pbar.update(1)
        # pbar.close()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm_stats = {}
        for name, model in models.items():
            norm_stats[name+'_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm = sum(norm_stats.values())

        # determine and set the learning rate for this iteration
        # lr = get_lr(step)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        # tokens_processed = train_loader.B * train_loader.T * grad_accum_steps *  ddp_world_size
        tokens_processed = 1
        tokens_per_sec = tokens_processed / dt

        if step % 1000 == 0:
            if master_process:
                writer.add_scalar("loss/train", loss_accum.item(), step)
                # writer.add_scalar("lr", lr, step)
                writer.add_scalar("norm/total", total_norm, step)
                for name, norm in norm_stats.items():
                    writer.add_scalar(f"norm/{name}", norm, step)
                writer.add_scalar("dt", dt, step)
                writer.add_scalar("tps", tokens_per_sec, step)

                print(f"step {step:5d} | loss: {loss_accum.item():.6f} | norm: {total_norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")

        if step %eval_every == 0:
            
            results = evaluate2(text_head, f_enc, b_enc, data_root = "tokenized_data", temperature = 1.0, top_k = 1, results=results, split = "val", max_batches = 100, B = 256, device=device)
            print(results)
            # generate(self, forward, backward_embedding, idx, goal, max_length, temperature=1.0, top_k = 1)
    
    if ddp:
        destroy_process_group()