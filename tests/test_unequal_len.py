import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1, 2, 3, 5, 6, 9, 3, 4, 5]).to(device)
b = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2]).to(device)

start = time.time()
for i in range(1000):
    final = []
    length = 4
    st = 0
    for i in range(b.max().int().item()+1):
        num = len((b == i).nonzero())
        curr = a[st:st+num]
        curr_zeros = torch.zeros(length-num).to(device)
        curr = torch.cat([curr, curr_zeros], dim=0)
        final.append(curr)
        st += num
    outs = torch.stack(final)
end = time.time()
print("Time elapsed using for loop:", end - start)
print(outs)

start = time.time()
for i in range(1000):
    L = len(a)
    rows = torch.nn.functional.one_hot(b).to(device)
    cols = rows.cumsum(0)[torch.arange(L).to(device), b] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs2 = (a[:, None, None] *
             cols[:, None, :] *
             rows[:, :, None]).sum(0)
end = time.time()
print("Time elapsed using matrix:", end - start)
print(outs2)
