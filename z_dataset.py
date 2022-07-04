from collections import defaultdict
from itertools import combinations
import pickle
import random
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
random.seed(555)

def preprocess_function(examples, tokenizer, baseline):
    assert len(examples["context"]) == len(examples["labels"])
    paragraphs = []
    for context, labels, q in zip(examples["context"], examples["labels"], examples["question"]):
        t1, t2 = context["title"][labels[0]], context["title"][labels[1]]
        sents = context["sentences"]
        p1 = f' {tokenizer.unk_token}'.join(context["sentences"][labels[0]])
        p1 = f'{q} {tokenizer.sep_token} {t1}: {tokenizer.unk_token} {p1} {tokenizer.sep_token}'
        if baseline:
            p2 = f' {tokenizer.unk_token}'.join(context["sentences"][labels[1]])
            p2 = f'{q} {tokenizer.sep_token} {t2}: {tokenizer.unk_token} {p2} {tokenizer.sep_token}'
            paragraphs += [p1, p2]
        else:
            p2 = f' {tokenizer.unk_token}'.join(context["sentences"][labels[1]])
            p2 = f'{t2}: {tokenizer.unk_token} {p2}'
            p = f"{p1} {p2}"
            paragraphs.append(p)
    tokenized_paras = tokenizer(paragraphs, truncation=True)['input_ids']
    if baseline:
        tokenized_paras = [tokenized_paras[i:i+2] for i in range(0, len(tokenized_paras), 2)]
    #tokenized_paras = tokenizer(paragraphs)
    #lengths = [len(para) for para in tokenized_paras['input_ids'] if len(para) > 512]
    return tokenized_paras

def sentence_level_preprocess_function(examples, tokenizer):
    paragraphs = [[c["sentences"][l[0]], c["sentences"][l[1]]] for c, l in zip(examples["context"], examples["labels"])]
    paragraphs = [p for ps in paragraphs for p in ps]
    lengths = [len(p) for p in paragraphs]
    questions = [[examples["question"][i//2]] * sum(lengths[i:i+2]) for i in range(0, len(lengths), 2)]
    lengths.insert(0, 0)
    for i in range(1, len(lengths)):
        lengths[i] += lengths[i-1]
    paragraphs = [p for ps in paragraphs for p in ps]
    questions = [q for qs in questions for q in qs]
    tokenized_paras = tokenizer(questions, paragraphs, truncation=True)['input_ids']
    tokenized_paras = [tokenized_paras[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    tokenized_paras = [tokenized_paras[i:i+2] for i in range(0, len(tokenized_paras), 2)]
    return tokenized_paras

def prepare(path, split, baseline=False):
    print("preparing HotpotQA")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    data = load_dataset('hotpot_qa', 'distractor')[split][:]

    labels = []
    sent_labels = []
    #supp_cnt = defaultdict(int)
    #supp_cnt2 = defaultdict(int)
    #sent_cnt = defaultdict(int)
    for title_label, titles in zip(data['supporting_facts'], data['context']):
        # dataset analysis
        #for sents in titles["sentences"]:
        #    sent_cnt[len(sents)] += 1
        #supp_cnt[len(title_label["title"])] += 1
        #tmp = list(set(title_label["title"]))
        #name1, name2 = tmp[0], tmp[1]
        #tmp = {name1: 0, name2: 0}
        #for t in title_label["title"]:
        #    tmp[t] += 1
        #supp_cnt2[tmp[name1]] += 1
        #supp_cnt2[tmp[name2]] += 1

        l = []
        for i in range(len(titles["title"])):
            if titles["title"][i] in title_label["title"]:
                l.append(i)
        assert len(l) == 2
        assert l[0] < l[1]
        labels.append(l)
        name1, name2 = titles["title"][l[0]], titles["title"][l[1]]
        tmp = {name1: [], name2: []}
        for t, sid in zip(title_label["title"], title_label["sent_id"]):
            tmp[t].append(sid)
        sls = [0] * (len(titles['sentences'][l[0]])+len(titles['sentences'][l[1]]))
        for item in tmp[name1]:
            if item >= len(titles['sentences'][l[0]]):
                print("INDEX OUT OF RANGE", item, len(titles['sentences'][l[0]]))
                continue
            sls[item] = 1
        for item in tmp[name2]:
            if item >= len(titles['sentences'][l[1]]):
                print("INDEX OUT OF RANGE", item, len(titles['sentences'][l[1]]))
                continue
            sls[item+len(titles['sentences'][l[0]])] = 1
        sent_labels.append(sls)
    data["labels"] = labels

    #for key in sorted(supp_cnt.keys()):
    #    print(f"{key} supp facts:", supp_cnt[key])
    #for key in sorted(supp_cnt2.keys()):
    #    print(f"{key} supp facts per paragraph:", supp_cnt2[key])
    #for key in sorted(sent_cnt.keys()):
    #    print(f"{key} sentences in a paragraph:", sent_cnt[key])
    if baseline:
        name = f"cache/hotpotqa_baseline_supp_encodings.pkl"
    else:
        name = f"cache/hotpotqa_supp_encodings.pkl"
    if split == "train":
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = preprocess_function(data, tokenizer, baseline)
            with open(name, 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = preprocess_function(data, tokenizer, baseline)
    if split == "train": 
        order = list(range(len(labels)))
        random.shuffle(order)
        num = int(0.9*len(order))
        train_indices = order[:num]
        valid_indices = order[num:]
        train_paras = [paras[i] for i in train_indices]
        valid_paras = [paras[i] for i in valid_indices]
        train_labels = [sent_labels[i] for i in train_indices]
        valid_labels = [sent_labels[i] for i in valid_indices]
        return (train_paras, valid_paras), (train_labels, valid_labels)
    else:
        return paras, sent_labels

def get_index(ref, ls):
    res = []
    for item in ref:
        if item >= len(ls):
            print("INDEX OUT OF RANGE", item, len(ls))
            continue
        res.append(item)
    rang = range(len(ls))
    combs = list(combinations(rang, r=1)) + list(combinations(rang, r=2)) + list(combinations(rang, r=3))
    try:
        idx = combs.index(tuple(res))
    except ValueError:
        idx = len(combs)
    return idx

def sentence_level_prepare(path, split, baseline=False):
    print("preparing HotpotQA")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    data = load_dataset('hotpot_qa', 'distractor')[split][:]

    labels = []
    sent_labels = []
    for title_label, titles in zip(data['supporting_facts'], data['context']):
        l = []
        for i in range(len(titles["title"])):
            if titles["title"][i] in title_label["title"]:
                l.append(i)
        assert len(l) == 2
        assert l[0] < l[1]
        labels.append(l)
        name1, name2 = titles["title"][l[0]], titles["title"][l[1]]
        tmp = {name1: [], name2: []}
        for t, sid in zip(title_label["title"], title_label["sent_id"]):
            tmp[t].append(sid)
        idx1 = get_index(tmp[name1], titles['sentences'][l[0]])
        idx2 = get_index(tmp[name2], titles['sentences'][l[1]])
        if baseline:
            sent_labels.append([idx1, idx2])
        else:
            if idx1 == len(titles['sentences'][l[0]]) or idx2 == len(titles['sentences'][l[1]]):
                sent_labels.append(idx1*idx2)
            else:
                sent_labels.append(idx1*len(titles['sentences'][l[0]])+idx2)
    data["labels"] = labels

    name = f"cache/hotpotqa_sent_supp_encodings.pkl"
    if split == "train":
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = sentence_level_preprocess_function(data, tokenizer)
            with open(name, 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = sentence_level_preprocess_function(data, tokenizer)
    if split == "train":
        order = list(range(len(labels)))
        random.shuffle(order)
        num = int(0.9*len(order))
        train_indices = order[:num]
        valid_indices = order[num:]
        train_paras = [paras[i] for i in train_indices]
        valid_paras = [paras[i] for i in valid_indices]
        train_labels = [sent_labels[i] for i in train_indices]
        valid_labels = [sent_labels[i] for i in valid_indices]
        return (train_paras, valid_paras), (train_labels, valid_labels)
    else:
        return paras, sent_labels

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
