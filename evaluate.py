import torch
from tqdm import tqdm
# from torch.utils.data import DataLoader
import config
from tokenizer import Tokenizer
import os
import numpy as np
# from train_gpt2 import load_tokens

# from utils.training_utils import AverageMeter

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class AverageMeter:
    def __init__(self):
        self.num = 0
        self.val = 0

    def update(self, val, num):
        self.val += val * num
        self.num += num

    def get(self, percentage=False):
        val = self.val / self.num * 100 if percentage else self.val / self.num
        return val

# Function to evaluate performance when generating
@torch.no_grad()
def evaluate(model, data_root = "tokenized_data", temperature = 1.0, top_k = 1, results=None, split = "val", max_batches = 10, B = 256, device="cpu"):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    # loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        assert split in {'val', 'train'}

        num_prefix_tokens = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3
        num_target_tokens = config.lenOfEachPath

        model.eval()
        total_acc = AverageMeter()
        tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
        
        tokenizer = Tokenizer(config.numOfPathsFromSource, config.lenOfEachPath, config.maxNodes)
        
        T = num_prefix_tokens + num_target_tokens

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        
        batch_idx = 0
        current_shard = 0
        current_token = 0
        
        tokens = load_tokens(shards[current_shard])
        tokens = tokens.to(device)
        # print(device)
        # print(tokens.device)
        # print(model.device)
        while batch_idx<max_batches:
            if current_token + B*T <= config.shard_size:
                batch_tokens = tokens[current_token : current_token + B*T].clone().view(B, T)
                x = batch_tokens.clone()[:, :-num_target_tokens]
                y = batch_tokens.clone()[:, -num_target_tokens:]
                # x.to(device)
                # y.to(device)
                # print(x.device, y.device)
                # generate(self, idx, max_length, temperature= 1.0, top_k = 1)
                y_pred = model.generate(x, T, temperature, top_k)
                # print(y_pred)
                correct = y.eq(y_pred[:, -num_target_tokens:]).float()
                # print(correct)
                completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
                # print(completely_correct)
                total_acc.update(completely_correct.item(), x.shape[0])

                # Individual token accuracy
                per_token_acc = correct.mean(dim=0)
                for i in range(num_target_tokens):
                    tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])
                
                # print(f'{split} accuracy: {total_acc.get(percentage=True):.2f}')
            else:
                current_shard = (current_shard + 1) % len(shards)
                current_token = 0
            batch_idx += 1
        
        if results is not None:
            results[split + '/accuracy'] = total_acc.get(percentage=True)
            for i in range(num_target_tokens):
                results[split + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
        model.train()
        return results
    
    # for x in bar:
    #     y = x[:, num_prefix_tokens:].clone()
    #     x = x[:, :num_prefix_tokens].clone()

    #     with ctx:
    #         y_pred = model.generate(x, num_target_tokens, temperature=temperature, top_k=top_k)
    #     #model.reset_cache()

    #     # Check how many tokens we get right and how many predictions are completely correct
    #     correct = y.eq(y_pred[:, -num_target_tokens:]).float()

    #     # Completely correct
    #     completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
    #     total_acc.update(completely_correct.item(), x.shape[0])

    #     # Individual token accuracy
    #     per_token_acc = correct.mean(dim=0)
    #     for i in range(num_target_tokens):
    #         tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])

    #     bar.set_description(f'{mode} accuracy: {total_acc.get(percentage=True):.2f}')

    # #model.empty_cache()

    # # Switch back to train mode
    # loader.dataset.train()
    # model.train()




# # Function to evaluate performance when applying teacher forcing
# @torch.no_grad()
# def evaluate_forced(model, loader, ctx, results=None, mode='test'):
#     """
#     Generates sequences with teacher-forcing and calculates accuracies
#     """
#     num_target_tokens = loader.dataset.num_target_tokens
#     total_acc, total_loss = AverageMeter(), AverageMeter()
#     tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
#     bar = tqdm(loader)

#     for x, y in bar:
#         # Produce logits with teacher-forcing (i.e. like during training)
#         with ctx:
#             logits, loss, accs = model(x, y)

#         total_acc.update(val=accs['acc'], num=x.shape[0])
#         total_loss.update(val=loss, num=x.shape[0])
#         for i in range(num_target_tokens):
#             tokens_corr[i].update(accs['token_acc'], x.shape[0])

#         bar.set_description('Forced Loss: {:.4f} Forced Acc: {:.2f}'.format(total_loss.get(),
#                                                               total_acc.get(percentage=True)))

#     if results is not None:
#         results[mode + '/forced loss'] = total_loss.get()
#         results[mode + '/forced accuracy'] = total_acc.get(percentage=True)
#         for i in range(num_target_tokens):
#             results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)

#     return results