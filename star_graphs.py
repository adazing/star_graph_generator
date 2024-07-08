"""
Tinystories dataset (for toy experimentation)
https://github.com/manantomar/tiny-tt/blob/master/Untitled.ipynb
Downloads and tokenizes the data and saves data shards to disk.
"""
import os
import multiprocessing as mp
import numpy as np
import tiktoken
# from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from tokenizer import Tokenizer
import config

T = config.numOfPathsFromSource * (config.lenOfEachPath - 1) * 3 + 3 + config.lenOfEachPath # sequence length


local_dir = "tokenized_data"
shard_size = T * int(1e4)


data = open("data.txt", "r").readlines()


# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)




# dataset = load_dataset('cyrilzhang/TinyStories2-ascii', split='train', download_mode="reuse_cache_if_exists")
# download the dataset
# ds = load_dataset('cyrilzhang/TinyStories2-ascii', split='train', download_mode="reuse_cache_if_exists")




# init the tokenizer
# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens['<|endoftext|>'] # end of text token
tokenizer = Tokenizer(config.numOfPathsFromSource, config.lenOfEachPath, config.maxNodes)


# eot = tokenizer.eot


def tokenize(line):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    # tokens = [eot] # the special <|endoftext|> token delimits all documents
    line = line.strip()
    prefix = line.split("=")[0] + "="
    target = line.split("=")[1]
    # tokens.extend(tokenizer.tokenize(prefix, target))
    tokens_np, number_of_tokens = tokenizer.tokenize(prefix, target)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    # print(tokens_np_uint16)
    # print(tokens_np_uint16)
    return tokens_np_uint16


# print(tokenize("8,13|6,44|30,41|8,7|13,6|31,28|8,36|7,31|4,16|36,4|41,24|8,30/8,16=8,36,4,16"))


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == "__main__":
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, data, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"tokenized_data_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                # print(all_tokens_np)
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder


        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"tokenized_data_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


