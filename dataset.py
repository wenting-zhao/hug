from collections import defaultdict
from itertools import combinations
import pickle
import random
import os
import torch


def split_data(*lists):
    order = list(range(len(lists[-1])))
    random.seed(555)
    random.shuffle(order)
    num = int(0.9*len(order))
    res = []
    for l in lists:
        if l is None:
            res.append(None)
        else:
            train_indices = order[:num]
            valid_indices = order[num:]
            x = [l[i] for i in train_indices]
            y = [l[i] for i in valid_indices]
            res.append((x, y))
    return res

def correct_format(ps):
    if len(ps) < 10:
        out = ps + ["none"] * (10 - len(ps))
    elif len(ps) > 10:
        out = ps[:10]
    return out

def preprocess_paragraph_function(examples, tokenizer, max_sent):
    paragraphs = [[s[:max_sent] for s in i["sentences"]] for i in examples["context"]]
    paragraphs = [correct_format(ps) if len(ps) != 10 else ps for ps in paragraphs]
    paragraphs = [[' '.join(s) for s in i] for i in paragraphs]
    questions = [[q] * 10 for i, q in enumerate(examples["question"])]
    paragraphs = [p for ps in paragraphs for p in ps]
    questions = [q for qs in questions for q in qs]
    tokenized_paras = tokenizer(questions, paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_paras = [tokenized_paras[i:i+10] for i in range(0, len(tokenized_paras), 10)]
    return tokenized_paras

def prepare_paragraphs(tokenizer, split, data, max_sent, baseline=False, no_x=False):
    print("preparing paragraphs")
    data = data[split]

    if not no_x:
        if split == "train":
            if os.path.isfile(f"cache/hotpotqa_encodings.pkl"):
                with open(f"cache/hotpotqa_encodings.pkl", 'rb') as f:
                    paras = pickle.load(f)
            else:
                paras = preprocess_paragraph_function(data, tokenizer, max_sent)
                with open(f"cache/hotpotqa_encodings.pkl", 'wb') as f:
                    pickle.dump(paras, f)
        else:
            paras = preprocess_paragraph_function(data, tokenizer, max_sent)
    else:
        paras = None

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

def preprocess_sentence_function(examples, tokenizer, baseline, unsup, max_sent):
    assert len(examples["context"]) == len(examples["labels"])
    paragraphs = []
    if unsup:
        for context, q in zip(examples["context"], examples["question"]):
            ts = context["title"]
            sents = context["sentences"]
            if len(ts) != 10:
                ts = correct_format(ts)
                sents = correct_format(sents)
            ps = []
            for i in range(len(sents)):
                t = ts[i]
                p = f' {tokenizer.unk_token}'.join(sents[i][:max_sent])
                p = f'{q} {tokenizer.sep_token} {t}: {tokenizer.unk_token} {p} {tokenizer.sep_token}'
                ps.append(p)
            paragraphs += ps
        tokenized_paras = tokenizer(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
        tokenized_paras = [tokenized_paras[i:i+10] for i in range(0, len(tokenized_paras), 10)]
    else:
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
        tokenized_paras = tokenizer(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
        if baseline:
            tokenized_paras = [tokenized_paras[i:i+2] for i in range(0, len(tokenized_paras), 2)]
        #tokenized_paras = tokenizer(paragraphs, return_attention_mask=False)
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
    tokenized_paras = tokenizer(questions, paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_paras = [tokenized_paras[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    tokenized_paras = [tokenized_paras[i:i+2] for i in range(0, len(tokenized_paras), 2)]
    return tokenized_paras

def prepare_sentences(tokenizer, split, data, max_sent, baseline=False, unsup=False):
    assert (baseline is False) or (unsup is False)
    print("preparing sentences")
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
    elif unsup:
        name = f"cache/hotpotqa_all_supp_encodings.pkl"
    else:
        name = f"cache/hotpotqa_supp_encodings.pkl"

    if split == "train":
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                paras = pickle.load(f)
        else:
            paras = preprocess_sentence_function(data, tokenizer, baseline, unsup, max_sent)
            with open(name, 'wb') as f:
                pickle.dump(paras, f)
    else:
         paras = preprocess_sentence_function(data, tokenizer, baseline, unsup, max_sent)
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
    print("preparing individual sentences")
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

def preprocess_answer_function(examples, tokenizer, max_sent):
    lengths = []
    sents = []
    contexts = [c for c in examples['context']]
    for i in range(len(contexts)):
        if len(contexts[i]['sentences']) != 10:
            contexts[i]['sentences'] = correct_format(contexts[i]['sentences'])
        for j in range(len(contexts[i]['sentences'])):
            curr = contexts[i]['sentences'][j][:max_sent]
            sents += curr
            lengths.append(len(curr))
    answers = [a for a in examples['answer']]
    questions = [q for q in examples['question']]
    lengths.insert(0, 0)
    for i in range(1, len(lengths)):
        lengths[i] += lengths[i-1]
    tokenized_sents = tokenizer(sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_answers = tokenizer(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_questions = tokenizer(questions, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    tokenized_sents = [tokenized_sents[i:i+10] for i in range(0, len(tokenized_sents), 10)]
    assert len(tokenized_sents) == len(tokenized_answers) == len(tokenized_questions)
    final = [(x, y) for x, y in zip(tokenized_questions, tokenized_sents)]
    return final, tokenized_answers

def prepare_answers(tokenizer, split, data, max_sent, baseline=False):
    print("preparing answers")
    data = data[split]

    if split == "train":
        if os.path.isfile(f"cache/hotpotqa_answer_encodings.pkl"):
            with open(f"cache/hotpotqa_answer_encodings.pkl", 'rb') as f:
                sents, answers = pickle.load(f)
        else:
            sents, answers = preprocess_answer_function(data, tokenizer, max_sent)
            with open(f"cache/hotpotqa_answer_encodings.pkl", 'wb') as f:
                pickle.dump((sents, answers), f)
    else:
        sents, answers = preprocess_answer_function(data, tokenizer, max_sent)
    if split == "train": 
        sents, answers = split_data(sents, answers)
    return sents, answers

def prepare_pipeline(tokenizer, answer_tokenizer, data, max_sent, para_ind=False, sent_ind=True):
    print("preparing HotpotQA")
    out = dict()
    out["train"] = dict()
    out["valid"] = dict()
    out["test"] = dict()
    _, para_labels = prepare_paragraphs(tokenizer, "train", data, max_sent=max_sent, no_x=True)
    paras, sent_labels = prepare_sentences(tokenizer, "train", data, max_sent=max_sent, unsup=True)
    supps, answ_labels = prepare_answers(answer_tokenizer, "train", data, max_sent=max_sent)
    assert len(paras) == len(para_labels) == len(answ_labels)
    out["train"]["paras"] = paras[0]
    out["valid"]["paras"] = paras[1]
    out["train"]["para_labels"] = para_labels[0]
    out["valid"]["para_labels"] = para_labels[1]
    out["train"]["sent_labels"] = sent_labels[0]
    out["valid"]["sent_labels"] = sent_labels[1]
    out["train"]["supps"] = supps[0]
    out["valid"]["supps"] = supps[1]
    out["train"]["answ_labels"] = answ_labels[0]
    out["valid"]["answ_labels"] = answ_labels[1]
    _, out["test"]["para_labels"] = prepare_paragraphs(tokenizer, "validation", data, max_sent=max_sent, no_x=True)
    out["test"]["paras"], out["test"]["sent_labels"] = prepare_sentences(tokenizer, "validation", data, max_sent=max_sent, unsup=True)
    out["test"]["supps"], out["test"]["answ_labels"] = prepare_answers(answer_tokenizer, "validation", data, max_sent=max_sent)
    return out

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

class UnsupHotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.paras = data["paras"]
        self.plabels = data["para_labels"]
        self.slabels = data["sent_labels"]
        self.supps = data["supps"]
        self.answs = data["answ_labels"]

    def __getitem__(self, idx):
        item = dict()
        item["paras"] = self.paras[idx]
        item["para_labels"] = self.plabels[idx]
        item["sent_labels"] = self.slabels[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        return item

    def __len__(self):
        return len(self.plabels)
