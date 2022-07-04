import pickle
import random
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
random.seed(555)

def preprocess_function(examples, tokenizer):
    paragraphs = [[s for s in i["sentences"]] for i in examples["context"]]
    paragraphs = [[' '.join(s) for s in i] for i in paragraphs]
    questions = [[q] * 10 for i, q in enumerate(examples["question"]) if len(paragraphs[i]) == 10]
    paragraphs = [p for p in paragraphs if len(p) == 10]
    paragraphs = sum(paragraphs, [])
    questions = sum(questions, [])
    tokenized_paras = tokenizer(questions, paragraphs, truncation=True)['input_ids']
    tokenized_paras = [tokenized_paras[i:i+10] for i in range(0, len(tokenized_paras), 10)]
    return tokenized_paras

def prepare(path, split):
    print("preparing HotpotQA")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    data = load_dataset('hotpot_qa', 'distractor')[split]

    if split == "train":
        if os.path.isfile(f"cache/hotpotqa_encodings.pkl"):
            with open(f"cache/hotpotqa_encodings.pkl", 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = preprocess_function(data, tokenizer)
            with open(f"cache/hotpotqa_encodings.pkl", 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = preprocess_function(data, tokenizer)
    ij2label = dict()
    cnt = 0
    for i in range(10):
        for j in range(i+1, 10):
            ij2label[(i, j)] = cnt
            cnt += 1
    labels = []
    for title_label, titles in zip(data['supporting_facts'], data['context']):
        title_label = title_label["title"]
        titles = titles["title"]
        if len(titles) != 10: continue
        l = []
        for i in range(len(titles)):
            if titles[i] in title_label:
                l.append(i)
        assert len(l) == 2
        assert l[0] < l[1]
        if baseline:
            label = [0] * 10
            label[l[0]] = 1
            label[l[1]] = 1
            labels.append(label)
        else:
            labels.append(ij2label[(l[0], l[1])])
    if split == "train": 
        order = list(range(len(labels)))
        random.shuffle(order)
        num = int(0.9*len(order))
        train_indices = order[:num]
        valid_indices = order[num:]
        train_paras = [paras[i] for i in train_indices]
        valid_paras = [paras[i] for i in valid_indices]
        train_labels = [labels[i] for i in train_indices]
        valid_labels = [labels[i] for i in valid_indices]
        return (train_paras, valid_paras), (train_labels, valid_labels)
    else:
        return paras, labels

class HotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, paras, labels):
        self.paras, self.labels = paras, labels

    def __getitem__(self, idx):
        item = dict()
        item['paras'] = self.paras[idx]
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
