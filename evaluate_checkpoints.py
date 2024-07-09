"""
1. Load the model checkpoint, either GPT2, or FB model.
2. Load the evaluation dataset. 
3. Evaluate the models' sequence generation performance on the evaluation dataset.

For evaluation of the forward backward model, we need:
- Conditional evaluation: conditioned on a goal text, evaluate the likelihood
- Unconditional evaluation: conditioned on no goal text, evaluate the likelihood

For comparing the FB model with GPT2, we would compare their 1-step prediction losses where the FB model gets to condition on the goal text, while GPT2 does not.
"""

import torch
from train_gpt2 import GPT, GPTConfig, DataLoaderLite, load_tokens
import config

class EvaluationDataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # get the shard filenames
        data_root = "tokenized_data"
        shards = os.listdir(data_root)
        shard = [s for s in shards if "val" in s][0]
        shard = os.path.join(data_root, shard)
        self.shard = shard
        if master_process:
            print(f"found 1 shard for split evaluation")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        x = torch.empty(size = (B*config.lenOfEachPath, T-1))
        # print(self.tokens.shape)
        if self.current_position+(B * config.lenOfEachPath * T * self.num_processes) < len(self.tokens): # wouldn't go over
            for idx in range(config.lenOfEachPath):
                buf = self.tokens[self.current_position : self.current_position+B*T]
                # print(buf.shape)
                x[idx*B:(idx + 1)*B, :] = buf.clone().view(B, T)[:, :-1] # inputs
                y[:B, :] = buf.clone().view(B, T)[:, 1:] # targets
                # print(x[0, :])
                y[:, :-config.lenOfEachPath] = config.maxNodes + 4 # empty
                # advance the position in the tensor
                self.current_position += B * T * self.num_processes
                # if loading the next batch would be out of bounds, advance to next shard
                if self.current_position + (B * T * self.num_processes) > len(self.tokens):
                    self.current_shard = (self.current_shard + 1) % len(self.shards)
                    self.tokens = load_tokens(self.shards[self.current_shard])
                    self.current_position = B * T * self.process_rank
                return x, y


model = GPT(GPTConfig(vocab_size=config.maxNodes + 5))

checkpoint = torch.load("/home/ada/Documents/dev/star_graphs/star_graph_generator/log/model_00999.pt")

model_state_dict = checkpoint['model']

model.load_state_dict(model_state_dict)

# VAL_PATH = '/home/ada/Documents/dev/star_graphs/star_graph_generator/tokenized_data/tokenized_data_val_000000.npy'

# print(VAL_PATH)

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


B = 256
T = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3 + config.lenOfEachPath # sequence length

val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

model.eval()

val_loader.reset()

with torch.no_grad():
    
