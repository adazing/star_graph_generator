import torch
import random
 
device_type='cuda'
 
def create_cartesian_product_examples(T):
    # precompute timestep pairs (t, t+k) for the forward and backward encoders
    ft = torch.arange(T, dtype=torch.int32).cuda() # [0, 1 .... T-1]
    k_rand = 5#random.randint(3, 20) # or sample from a logarithimic
    dt = torch.Tensor([2, k_rand]).type(torch.int32).to(device_type)
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
 
    return fb_pairs, labels, fb_pairs[:,1] - fb_pairs[:, 0], midpoints
 
 
pairs, labels, ks, midpoints = create_cartesian_product_examples(10)
 
print(pairs)
print(labels)
print(ks)
print(midpoints)
