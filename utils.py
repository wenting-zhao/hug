import torch

def padding(indices, values):
    L = len(indices)
    rows = torch.nn.functional.one_hot(indices)
    cols = rows.cumsum(0)[torch.arange(L), indices] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (values[:, None, None] *
            cols[:, None, :] *
            rows[:, :, None]).sum(0)
    return outs
