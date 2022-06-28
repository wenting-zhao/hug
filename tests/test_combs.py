import torch
import time
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bs = 16
num_choices = 10
emb_size = 128
sentence_embeddings = torch.rand(bs, num_choices, emb_size).to(device)
linear = nn.Linear(emb_size, 1)
linear.to(device)
mlp = nn.Sequential(
        nn.Linear(emb_size*2, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
        )
mlp.to(device)
m = nn.Softmax(dim=-1)

single_outs = linear(sentence_embeddings).view(bs, -1)
single_outs = m(single_outs)

start = time.time()
for i in range(100):
    combs = torch.cartesian_prod(torch.arange(num_choices), torch.arange(num_choices))
    C = len(combs)
    paired = sentence_embeddings[:,combs,:]
    pairs = paired.view(bs,C,-1)
    pair_outs = mlp(pairs).view(bs, -1)
    pair_outs = m(pair_outs).reshape(bs, num_choices, num_choices)
    pair_outs = pair_outs.permute(2, 0, 1)
    outs = single_outs * pair_outs
    outs = outs.permute(1, 2, 0)
    outs = outs + outs.permute(0, 2, 1)
    indices = torch.triu_indices(num_choices, num_choices, offset=1).to(device)
    outs = outs[:, indices[0], indices[1]]
end = time.time()
print("Time elapsed using matrix:", end - start)

start = time.time()
for i in range(100):
    pairs2 = []
    for emb in sentence_embeddings:
        for i in range(num_choices):
            for j in range(num_choices):
                pairs2.append(torch.cat([emb[i], emb[j]]))
    pairs2 = torch.stack(pairs2).view(bs, num_choices, num_choices, -1).to(device)
    pair_outs2 = mlp(pairs2).view(bs, -1)
    pair_outs2 = m(pair_outs2).view(bs, num_choices, num_choices)
    for b in range(bs):
        for i in range(pair_outs2.shape[1]):
            pair_outs2[b][i] = single_outs[b][i] * pair_outs2[b][i]
    outs2 = []
    for b in range(bs):
        intermediate = []
        for i in range(num_choices):
            for j in range(i+1, num_choices):
                tmp = pair_outs2[b, i, j] + pair_outs2[b, j, i]
                tmp = tmp.item()
                intermediate.append(tmp)
        outs2.append(intermediate)
    outs2 = torch.Tensor(outs2).to(device)
end = time.time()
print(torch.allclose(outs, outs2))
print("Time elapsed using for loop:", end - start)
