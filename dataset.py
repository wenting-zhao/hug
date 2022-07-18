from collections import defaultdict
from itertools import combinations
import pickle
import random
import os
import torch
random.seed(555)


def split_data(*lists):
    order = list(range(len(lists[0])))
    random.shuffle(order)
    num = int(0.9*len(order))
    res = []
    for l in lists:
        train_indices = order[:num]
        valid_indices = order[num:]
        x = [l[i] for i in train_indices]
        y = [l[i] for i in valid_indices]
        res.append((x, y))
    return res

def preprocess_paragraph_function(examples, tokenizer):
    paragraphs = [[s for s in i["sentences"]] for i in examples["context"]]
    paragraphs = [[' '.join(s) for s in i] for i in paragraphs]
    questions = [[q] * 10 for i, q in enumerate(examples["question"]) if len(paragraphs[i]) == 10]
    paragraphs = [p for p in paragraphs if len(p) == 10]
    paragraphs = [p for ps in paragraphs for p in ps]
    questions = [q for qs in questions for q in qs]
    tokenized_paras = tokenizer(questions, paragraphs, truncation=True)['input_ids']
    tokenized_paras = [tokenized_paras[i:i+10] for i in range(0, len(tokenized_paras), 10)]
    return tokenized_paras

def prepare_paragraphs(tokenizer, split, data, baseline=False):
    print("preparing HotpotQA")
    data = data[split]

    if split == "train":
        if os.path.isfile(f"cache/hotpotqa_encodings.pkl"):
            with open(f"cache/hotpotqa_encodings.pkl", 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = preprocess_paragraph_function(data, tokenizer)
            with open(f"cache/hotpotqa_encodings.pkl", 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = preprocess_paragraph_function(data, tokenizer)
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
        paras, labels = split_data(paras, labels)
    return paras, labels

def preprocess_sentence_function(examples, tokenizer, baseline):
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

def sentence_level_preprocess_function(examples, tokenizer, threshold):
    paragraphs = [[c["sentences"][l[0]], c["sentences"][l[1]]] for c, l in zip(examples["context"], examples["labels"])]
    paragraphs = [p[:threshold] for ps in paragraphs for p in ps]
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

def prepare_sentences(tokenizer, split, data, baseline=False):
    print("preparing HotpotQA")
    data = data[split][:]

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
            paras = preprocess_sentence_function(data, tokenizer, baseline)
            with open(name, 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = preprocess_sentence_function(data, tokenizer, baseline)
    if split == "train": 
        paras, sent_labels = split_data(paras, sent_labels)
    return paras, sent_labels

def get_index(ref, ls, threshold):
    res = []
    for item in ref:
        if item >= len(ls):
            print("INDEX OUT OF RANGE", item, len(ls))
            continue
        res.append(item)
    if len(ls) > threshold:
        ls = ls[:threshold]
    rang = range(len(ls))
    combs = list(combinations(rang, r=1)) + list(combinations(rang, r=2)) + list(combinations(rang, r=3))
    try:
        idx = (combs.index(tuple(res)), len(combs))
    except ValueError:
        idx = (len(combs), len(combs))
    return idx

def prepare_individual_sentences(tokenizer, split, data, baseline=False, threshold=10):
    print("preparing HotpotQA")
    data = data[split][:]

    labels = []
    sent_labels = []
    rang = range(threshold)
    max_combs = len(list(combinations(rang, r=1)) + list(combinations(rang, r=2)) + list(combinations(rang, r=3)))
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
        idx1 = get_index(tmp[name1], titles['sentences'][l[0]], threshold)
        idx2 = get_index(tmp[name2], titles['sentences'][l[1]], threshold)
        if baseline:
            sent_labels.append([idx1[0], idx2[0]])
        else:
            # first two conditions handle # of sentences in a paragraph > threshold
            # latter two conditions handle # of supporting facts > 3 in a paragraph
            if idx1[0] == max_combs or idx2[0] == max_combs or idx1[0] == idx1[1] or idx2[0] == idx2[1]:
                sent_labels.append(idx1[1]*idx2[1])
            else:
                sent_labels.append(idx1[0]*idx2[1]+idx2[0])
    data["labels"] = labels

    if baseline:
        name = f"cache/hotpotqa_baseline_sent_supp_encodings.pkl"
    else:
        name = f"cache/hotpotqa_sent_supp_encodings.pkl"
    if split == "train":
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = sentence_level_preprocess_function(data, tokenizer, threshold)
            with open(name, 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = sentence_level_preprocess_function(data, tokenizer, threshold)
    if split == "train": 
        paras, sent_labels = split_data(paras, sent_labels)
    return paras, sent_labels

def preprocess_answer_function(examples, tokenizer, threshold):
    lengths = []
    slengths = []
    sents = []
    indices = []
    for i in range(len(examples['context'])):
        curr_indices = dict()
        for j in range(len(examples['context'][i]['sentences'])):
            curr = examples['context'][i]['sentences'][j][:threshold]
            rang = range(len(curr))
            rang_combs = list(combinations(rang, r=1)) + list(combinations(rang, r=2)) + list(combinations(rang, r=3))
            curr_indices[j] = rang_combs
            combs = list(combinations(curr, r=1)) + list(combinations(curr, r=2)) + list(combinations(curr, r=3))
            combs = [examples["question"][i] + f' {tokenizer.sep_token} ' + ' '.join(c) for c in combs]
            sents += combs
            lengths.append(len(combs))
        slengths.append(j+1)
        indices.append(curr_indices)
    answers = [a for a in examples['answer']]
    lengths.insert(0, 0)
    for i in range(1, len(lengths)):
        lengths[i] += lengths[i-1]
    slengths.insert(0, 0)
    for i in range(1, len(slengths)):
        slengths[i] += slengths[i-1]
    tokenized_sents = tokenizer(sents, truncation=True)['input_ids']
    tokenized_answers = tokenizer(answers, truncation=True)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    tokenized_sents = [tokenized_sents[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    final = []
    for i, sent in zip(indices, tokenized_sents):
        curr = dict()
        for key in i.keys():
            sub_curr = dict()
            for j, subkey in enumerate(i[key]):
                sub_curr[subkey] = sent[key][j]
            curr[key] = sub_curr
        final.append(curr)
    assert len(final) == len(tokenized_sents)
    return final, tokenized_answers

def prepare_answers(tokenizer, split, data, threshold=10, baseline=False):
    print("preparing HotpotQA")
    data = data[split][:100]

    if split == "train":
        if os.path.isfile(f"cache/hotpotqa_answer_encodings.pkl"):
            with open(f"cache/hotpotqa_answer_encodings.pkl", 'rb') as f:
                sents, answers = pickle.load(f)
        else:
            sents, answers = preprocess_answer_function(data, tokenizer, threshold)
            with open(f"cache/hotpotqa_answer_encodings.pkl", 'wb') as f:
                pickle.dump((sents, answers), f)
    else:
        sents, answers = preprocess_answer_function(data, tokenizer, threshold)
    if split == "train": 
        sents, answers = split_data(sents, answers)
    return sents, answers

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
