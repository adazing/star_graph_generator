import os
import numpy as np
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']




maxNodes = 50


class Tokenizer:
    def __init__(self, numOfPathsFromSource, lenOfEachPath, maxNodes):
        self.numOfPathsFromSource = numOfPathsFromSource
        self.lenOfEachPath = lenOfEachPath
        self.maxNode = maxNodes
        self.encoder = {str(i): i for i in range(maxNodes)}
        self.encoder['|'] = maxNodes
        self.encoder['='] = maxNodes + 1
        self.encoder['/'] = maxNodes + 2
        self.encoder['$'] = maxNodes + 3
        # self.encoder[','] = maxNodes + 4
        # self.eot = maxNodes + 5


        self.decoder = {i: i for i in range(maxNodes)}
        self.decoder[maxNodes] = '|'
        self.decoder[maxNodes + 1] = '='
        self.decoder[maxNodes + 2] = '/'
        self.decoder[maxNodes + 3] = '$'
        # self.decoder[maxNodes + 4] = ','
        self.decoder[maxNodes + 4] = ''
        self.decoder[-1] = ':'
   
    def encode(self, data):
       
        out = []
        i = 0
        while i < len(data):
            if data[i] == ',':
                i += 1
                continue
            s = ''
            j = 0
            while i + j < len(data) and data[i + j] in numbers:
                s += data[i + j]
                j += 1
            if s == '':
                s = data[i]
                i += 1
            else:
                i += j
            out.append(self.encoder[s])
        # print(out)
        return out
   
    def decode(self, data):
        return [self.decoder[i] for i in data]
   
    def tokenize(self, prefix, target):
        '''
            takes line of data
        '''
        # out = [eot]
        prefix_len = len(self.encode(prefix))
        target_len = len(self.encode(target))
        # same_len = True
        # for prefix, target in data_list:
        prefix = np.asarray(self.encode(prefix))
        target = np.asarray(self.encode(target))
        seq = np.concatenate([prefix, target])
        # out.append(seq)


        # Check if all prefixes and all targets have the same length
        return seq, prefix_len + target_len + 1


# tokenizer = tokenize()


# dir_path = os.path.join(os.path.dirname(__file__), "tokenized_data")


# shard_size = int(1e7) # 100M tokens per shard, total of 100 shards
# token_count = 0


# lines = open("data.txt", "r").readlines()
# num_of_lines = len(lines)


# tokenizer = Tokenizer(4, 4, maxNodes)


# split="train"
# shard_index = 0


# # os.mkdir(dir_path)


# # while True:
# all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
# # eot = maxNodes + 4
# for l in range(num_of_lines):
#     numberOfRectangles = int((l+1)*50/num_of_lines)
#     bar = '█'*numberOfRectangles + " "*(50-numberOfRectangles)
#     print(f'\r|{bar}| {(l+1)*100/num_of_lines:.1f}%', end="", flush=True)
   
#     prefix, target = lines[l].strip().split("=")
#     prefix += "="
#     # print(prefix)
#     tokens, size = tokenizer.tokenize(prefix, target)
#     # print(tokens, size)
#     # print(token_count)


#     if token_count+size<shard_size:
#         all_tokens_np[token_count:token_count+size] = tokens
#         token_count += size
#     else:
#         shard_index += 1
#         filename = os.path.join(dir_path, f"data_{split}_{shard_index:06d}")
#         np.save(filename, all_tokens_np)
#         token_count = 0
#         all_tokens_np = np.empty((shard_size,), dtype=np.uint16)


# if token_count!=0:
#     filename = os.path.join(dir_path, f"data_{split}_{shard_index:06d}")
#     np.save(filename, all_tokens_np)
#     token_count = 0
#     all_tokens_np = np.empty((shard_size,), dtype=np.uint16)


