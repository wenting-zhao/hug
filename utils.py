from datasets import load_dataset
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hotpotqa():
    return load_dataset('hotpot_qa', 'distractor')

def padding(indices, values):
    L = len(indices)
    rows = torch.nn.functional.one_hot(indices)
    cols = rows.cumsum(0)[torch.arange(L), indices] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (values[:, None, None] *
            cols[:, None, :] *
            rows[:, :, None]).sum(0)
    return outs

def padding_long(indices, values):
    final = []
    length = max(indices)
    st = 0
    for i in indices:
        curr = values[st:st+i]
        curr_zeros = torch.zeros(length-i).to(device)
        curr = torch.cat([curr, curr_zeros], dim=0)
        final.append(curr)
        st += i
    outs = torch.stack(final)
    return outs
