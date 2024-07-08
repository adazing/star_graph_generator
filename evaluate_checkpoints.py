"""
1. Load the model checkpoint, either GPT2, or FB model.
2. Load the evaluation dataset. 
3. Evaluate the models' sequence generation performance on the evaluation dataset.

For evaluation of the forward backward model, we need:
- Conditional evaluation: conditioned on a goal text, evaluate the likelihood
- Unconditional evaluation: conditioned on no goal text, evaluate the likelihood

For comparing the FB model with GPT2, we would compare their 1-step prediction losses where the FB model gets to condition on the goal text, while GPT2 does not.
"""
from train_fb import Encoder, ForwardEncoderConfig, BackwardEncoderConfig, TextHead, TextHeadConfig,DataLoaderLite
import torch
from tqdm import tqdm

def load_fb_model():
    pass 

def load_gpt_model():
    pass

if __name__ == "__main__":
    device = "cuda"
    # load the model checkpoint
    checkpoint = torch.load("/home/t-edwardhu/build-nanogpt/log/2024-07-01-23-43-25/model_00999.pt")
    f_enc = Encoder(checkpoint['raw_f_enc_config'])
    b_enc = Encoder(checkpoint['raw_b_enc_config'])
    text_head = TextHead(TextHeadConfig(n_goal=b_enc.config.n_embd))
    use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        f_enc = torch.compile(f_enc)
        b_enc = torch.compile(b_enc)
        text_head = torch.compile(text_head)

    f_enc.load_state_dict(checkpoint["raw_f_enc"])
    b_enc.load_state_dict(checkpoint["raw_b_enc"])
    text_head.load_state_dict(checkpoint["raw_text_head"])
 
    models = [f_enc, b_enc, text_head]
    for m in models:
        m.to(device)
        m.eval()

    B = 64
    T = 1024
    k =1022 # offset between forward and backward representations.
    # we should consider evaluating all k <= K rather than just a single K as we do currently. We can use the cartesian product code for this.
    # TODO: we should modify the code so that for a given trajectory, we first take a goal with length K, and then evaluate the model for all valid starting points in the trajectory, conditioned on the goal.
    """ 
    k=0 case
        --f-><-k-><--b------
    X =  1  2  3  4  5  6  7
               y

    k=1 case
        --f-><---k--><--b------
    X =  1  2  3  4  5  6  7
               y      
    """
    val_loss_accum = 0
    val_loader = DataLoaderLite(B=B, T=T, split="val", process_rank=0, num_processes=1)
    torch.set_float32_matmul_precision('high')
    # calculate log probability of the generated sequences.
    with torch.no_grad():
        val_loss_steps = 5
        pbar = tqdm(range(val_loss_steps), desc=f"Calculating validation loss.", disable=False)
        # TODO: go through all the validation data.
        for _ in range(val_loss_steps):
            buf = val_loader.next_batch()
            buf = buf.to(device)
            x = buf[:-1].view(B, T)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                                # Let's say k=2 and the array is (1,2,3,4,5,6)
                forward = f_enc(x)[:, :-(1+k)]  # [f(1), f(12), f(123)]
                _backward = b_enc(x)            # [b(6), b(65), b(654), b(6543), b(65432), b(654321)]
                backward = _backward.flip(1)    # [b(6..1), b(6..2), b(6..3), b(6..4), b(6..5), b(6)]
                backward = backward[:, (1+k):]  # [b(654), b(65), b(6)]
                text_labels = x[:,1 : -(k)]     # [2,      3,     4]
                assert forward.shape[1] == backward.shape[1] == text_labels.shape[1]
                print(forward.shape, backward.shape, text_labels.shape)
                logits, loss = text_head(forward, backward, targets=text_labels)
                
                loss = loss / (val_loss_steps)
                val_loss_accum += loss.detach()
                pbar.update(1)
        pbar.close()
    print("Validation loss: ", val_loss_accum.item())