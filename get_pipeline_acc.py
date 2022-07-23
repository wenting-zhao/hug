import sys
from transformers import AutoTokenizer
import torch
from utils import load_hotpotqa
from dataset import prepare_paragraphs, prepare_sentences, prepare_individual_sentences

data = load_hotpotqa()
indices = []
for i, item in enumerate(data["validation"]['context']):
    if len(item['title']) != 10:
        indices.append(i)
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
_, labels = prepare_sentences(tokenizer, "validation", data)
_, labels2 = prepare_individual_sentences(tokenizer, "validation", data)
_, plabels = prepare_paragraphs(tokenizer, "validation", data)

results = torch.load(sys.argv[1])
t = float(sys.argv[2])
length = len(results[0])
labels = [labels[i:min(i+length, len(labels))] for i in range(0, len(labels), length)]
acc = []
for pred, ref in zip(results, labels):
    pred = pred.detach().clone()
    pred[pred>t] = 1
    pred[pred<=t] = 0
    for i, j in zip(pred, ref):
        i = i.tolist()[:len(j)]
        acc.append(i == j)
sacc = []
for i in range(len(acc)):
    if i not in indices:
        sacc.append(acc[i])

results2 = torch.load(sys.argv[3])
labels2 = [labels2[i:min(i+length, len(labels2))] for i in range(0, len(labels2), length)]
acc2 = []
for pred, ref in zip(results2, labels2):
    pred = pred.argmax(dim=-1)
    for i, j in zip(pred, ref):
        acc2.append(i.item() == j)
sacc2 = []
for i in range(len(acc2)):
    if i not in indices:
        sacc2.append(acc2[i])

presults = torch.load(sys.argv[4])
predictions = presults.argmax(dim=-1)
pacc = []
for pred, ref in zip(predictions, plabels):
    if pred.item() == ref:
        pacc.append(1)
    else:
        pacc.append(0)
assert len(pacc) == len(sacc) == len(sacc2)
final = [pacc[i] * sacc[i] for i in range(len(pacc))]
print(sum(final) / len(final))
final = [pacc[i] * sacc2[i] for i in range(len(pacc))]
print(sum(final) / len(final))
