from collections import defaultdict, Counter
from glob import glob
from itertools import combinations, product
import json
import pickle
import random
import os
import torch


def len_helper(l):
    l.insert(0, 0)
    for i in range(1, len(l)):
        l[i] += l[i-1]
    return l

def preprocess_hotpotqa_function(examples, tok, answ_tok, fixed, max_sent):
    tot_sents = []
    supps = []
    lengths = []
    slengths = []
    ds = []
    num_s = []
    sent_labels = []
    for context, labels, q, supp in zip(examples["context"], examples["labels"], examples["question"], examples['supporting_facts']):
        ts = context["title"]
        sents = context["sentences"]
        k = 0
        mapping = dict()
        for i in range(len(ts)):
            for j in range(len(sents[i])):
                mapping[(i, j)] = k
                k += 1
        # correct paragrpah indices
        x, y = labels
        norm_label = []
        for t, sid in zip(supp["title"], supp["sent_id"]):
            if t == ts[x]:
                if sid >= len(sents[x]):
                    print("INDEX OUT OF RANGE", sid, len(sents[x]))
                    continue
                norm_label.append(mapping[(x, sid)])
            else:
                if sid >= len(sents[y]):
                    print("INDEX OUT OF RANGE", sid, len(sents[y]))
                    continue
                norm_label.append(mapping[(y, sid)])
        sent_labels.append(norm_label)
        all_es = [ss for s in sents for ss in s]
        curr_sents = []
        for i in range(0, len(all_es), fixed):
            sent = f'{tok.unk_token} ' + f' {tok.unk_token} '.join(all_es[i:i+fixed])
            sent = q + f' {tok.sep_token} ' + sent
            curr_sents.append(sent)
        lengths.append(len(curr_sents))
        tot_sents += curr_sents
        z_len = len(all_es)
        rang = list(range(z_len))
        curr_idxes = []
        for i in range(1, max_sent+1):
            curr_idxes += list(combinations(rang, r=i))
        for idxes in curr_idxes:
            curr_supps = [all_es[m] for m in idxes]
            curr_supps = ' '.join(curr_supps)
            curr_supps = q + f' {answ_tok.sep_token} ' + curr_supps
            supps.append(curr_supps)
        ds.append(curr_idxes)
        slengths.append(len([x for xx in curr_idxes for x in xx]))
        num_s.append(z_len)
    lengths = len_helper(lengths)
    slengths = len_helper(slengths)
    tokenized_sents = tok(tot_sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    answers = [a for a in examples["answer"]]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s, sent_labels

def prepare_hotpotqa(tokenizer, answ_tokenizer, split, data, fixed, max_e, path=None):
    assert fixed > 0
    print("preparing hotpotqa")
    if split == "val": split = "validation"
    data = data[split][:]

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
        labels.append(l)
    data["labels"] = labels
    fname = f"cache/hotpotqa_rag_{split}.pkl"

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answers, ds, num_s, sent_labels = pickle.load(f)
    else:
        sents, supps, answers, ds, num_s, sent_labels = preprocess_hotpotqa_function(data, tokenizer, answ_tokenizer, fixed, max_e)
        with open(fname, 'wb') as f:
            pickle.dump((sents, supps, answers, ds, num_s, sent_labels), f)
    return (sents, supps, answers, ds, num_s, sent_labels, data["answer"])

class HotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        self.sents, self.supps, self.answs, self.ds, self.num_s, self.sent_labels, self.labels = everything

    def __getitem__(self, idx):
        item = dict()
        item['sents'] = self.sents[idx]
        item['supps'] = self.supps[idx]
        item['answs'] = self.answs[idx]
        item['ds'] = self.ds[idx]
        item['num_s'] = self.num_s[idx]
        item['sent_labels'] = self.sent_labels[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)

def preprocess_fever(examples, tok, answ_tok, fixed, max_e):
    sents = []
    supps = []
    lengths = []
    slengths = []
    ds = []
    num_s = []
    for e in examples:
        curr_sents = []
        for i in range(0, len(e['z']), fixed):
            sent = f'{tok.unk_token} ' + f' {tok.unk_token} '.join(e['z'][i:i+fixed])
            sent = e['x'] + f' {tok.sep_token} ' + sent
            curr_sents.append(sent)
        lengths.append(len(curr_sents))
        sents += curr_sents
        z_len = len(e['z'])
        rang = list(range(z_len))
        curr_idxes = []
        for i in range(1, max_e+1):
            curr_idxes += list(combinations(rang, r=i))
        for idxes in curr_idxes:
            curr_supps = [e['z'][m] for m in idxes]
            curr_supps = ' '.join(curr_supps)
            curr_supps = e['x'] + f' {answ_tok.sep_token} ' + curr_supps
            supps.append(curr_supps)
        ds.append(curr_idxes)
        slengths.append(len([x for xx in curr_idxes for x in xx]))
        num_s.append(z_len)
    lengths = len_helper(lengths)
    slengths = len_helper(slengths)
    # there is one example that has a length 529, might cause an error
    tokenized_sents = tok(sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    answers = [e['y'] for e in examples]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s

def prepare_fever(tokenizer, answer_tokenizer, split, docs, fixed, max_e, path="data/fever/"):
    print(f"prepare fever {split}")
    data = []
    with open(f"{path}/{split}.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    out = []
    labels = []
    sent_labels = []
    for d in data:
        curr = dict()
        curr['x'] = d['query']
        d['evidences'] = [ee for e in d['evidences'] for ee in e]
        docid = [l['docid'] for l in d['evidences']]
        docid = set(docid)
        assert len(docid) == 1
        docid = docid.pop()
        curr['z'] = docs[docid]
        gold_z = [l['start_sentence'] for l in d['evidences']]
        sent_labels.append(gold_z)
        curr['y'] = d['classification'].lower()
        label = 0 if curr['y'] == "supports" else 1
        labels.append(label)
        out.append(curr)
    fname = f"cache/fever_rag_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_fever(out, tokenizer, answer_tokenizer, fixed, max_e)
        with open(fname, 'wb') as f:
            pickle.dump((sents, supps, answs, ds, num_s), f)
    return (sents, supps, answs, ds, num_s, sent_labels, labels)

class FeverDataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        self.sents, self.supps, self.answs, self.ds, self.num_s, self.sent_labels, self.labels = everything

    def __getitem__(self, idx):
        item = dict()
        item['sents'] = self.sents[idx]
        item['supps'] = self.supps[idx]
        item['answs'] = self.answs[idx]
        item['ds'] = self.ds[idx]
        item['num_s'] = self.num_s[idx]
        item['sent_labels'] = self.sent_labels[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)

def preprocess_musique(examples, tok, answ_tok, fixed, max_e):
    sents = []
    supps = []
    lengths = []
    slengths = []
    ds = []
    num_s = []
    for e in examples:
        curr_sents = []
        for i in range(0, len(e['z']), fixed):
            sent = f'{tok.unk_token} ' + f' {tok.unk_token} '.join(e['z'][i:i+fixed])
            sent = e['x'] + f' {tok.sep_token} ' + sent
            curr_sents.append(sent)
        lengths.append(len(curr_sents))
        sents += curr_sents
        z_len = len(e['z'])
        rang = list(range(z_len))
        curr_idxes = []
        for i in range(1, max_e+1):
            curr_idxes += list(combinations(rang, r=i))
        for one in curr_idxes:
            curr_supps = [e['z'][m] for m in one]
            curr_supps = ' '.join(curr_supps)
            q = e['x']
            curr_supps = q + f' {answ_tok.sep_token} ' + curr_supps
            supps.append(curr_supps)
        ds.append(curr_idxes)
        slengths.append(len(curr_idxes))
        num_s.append(z_len)
    lengths = len_helper(lengths)
    slengths = len_helper(slengths)
    tokenized_sents = tok(sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    answers = [e['y'] for e in examples]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s

def prepare_musique(tokenizer, answer_tokenizer, split, docs, fixed, max_e, path="data/musique/"):
    print(f"prepare MuSiQue {split}")
    data = []
    with open(f"{path}/musique_ans_v1.0_{split}.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    out = []
    sent_labels = []
    labels = []
    for d in data:
        curr = dict()
        curr['x'] = d["question"]
        curr['y'] = d["answer"]
        curr['z'] = []
        sent_label = []
        for p in d["paragraphs"]:
            curr['z'].append(p["paragraph_text"])
            if p["is_supporting"]:
                sent_label.append(p["idx"])
        labels.append(d["answer"])
        sent_labels.append(sent_label)
        out.append(curr)
    fname = f"cache/musique_rag_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_musique(out, tokenizer, answer_tokenizer, fixed, max_e)
        with open(fname, 'wb') as f:
            pickle.dump((sents, supps, answs, ds, num_s), f)
    return (sents, supps, answs, ds, num_s, sent_labels, labels)

class MuSiQueDataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        self.sents, self.supps, self.answs, self.ds, self.num_s, self.sent_labels, self.labels = everything

    def __getitem__(self, idx):
        item = dict()
        item['sents'] = self.sents[idx]
        item['supps'] = self.supps[idx]
        item['answs'] = self.answs[idx]
        item['ds'] = self.ds[idx]
        item['num_s'] = self.num_s[idx]
        item['sent_labels'] = self.sent_labels[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)

def preprocess_multirc(examples, tok, answ_tok, fixed, max_e):
    sents = []
    supps = []
    lengths = []
    slengths = []
    ds = []
    num_s = []
    for e in examples:
        curr_sents = []
        for i in range(0, len(e['z']), fixed):
            sent = f'{tok.unk_token} ' + f' {tok.unk_token} '.join(e['z'][i:i+fixed])
            sent = e['x'] + f' {tok.sep_token} ' + sent
            curr_sents.append(sent)
        lengths.append(len(curr_sents))
        sents += curr_sents
        z_len = len(e['z'])
        rang = list(range(z_len))
        curr_idxes = []
        for i in range(1, max_e+1):
            curr_idxes += list(combinations(rang, r=i))
        for one in curr_idxes:
            curr_supps = [e['z'][m] for m in one]
            curr_supps = ' '.join(curr_supps)
            q = e['x']
            curr_supps = f'Question: {q}' + f' {answ_tok.sep_token} ' + curr_supps
            supps.append(curr_supps)
        ds.append(curr_idxes)
        slengths.append(len(curr_idxes))
        num_s.append(z_len)
    lengths = len_helper(lengths)
    slengths = len_helper(slengths)
    #tokenized_sents = tok(sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = tok(sents, return_attention_mask=False)['input_ids']
    for s in tokenized_sents:
        if len(s) > 512:
            print("WARNING")
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    answers = [e['y'] for e in examples]
    if not isinstance(answers[0], list):
        print("not list")
        tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    else:
        print("list")
        tokenized_answers = [answ_tok(a, truncation=True, return_attention_mask=False)['input_ids'] for a in answers]
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s

def prepare_multirc(tokenizer, answer_tokenizer, split, docs, fixed, max_e, path="data/multirc/"):
    print(f"prepare MultiRC {split}")
    data = []
    with open(f"{path}/{split}.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    out = []
    sent_labels = []
    labels = []
    groups = defaultdict(list)
    for d in data:
        idx = d["annotation_id"].rfind(':')
        key = d["annotation_id"][:idx].strip()
        groups[key].append(d)
    for _, values in groups.items():
        d = values[0]
        curr = dict()
        curr['x'] = d['query'].split('||')[0]
        d['evidences'] = [ee for e in d['evidences'] for ee in e]
        docid = [l['docid'] for l in d['evidences']]
        docid = set(docid)
        assert len(docid) == 1
        docid = docid.pop()
        curr['z'] = docs[docid]
        gold_z = [l['start_sentence'] for l in d['evidences']]
        sent_labels.append(gold_z)
        labels.append([0 if d['classification'] == 'True' else 1 for d in values])
        if split == "train":
            answers = [d['query'].split('||')[-1] + ' (correct)' if d['classification'] == 'True' else d['query'].split('||')[-1] + ' (wrong)' for d in values]
            curr['y'] = ['Answer:' + a for a in answers]
            curr['y'] = f' {answer_tokenizer.sep_token}'.join(answers)
        else:
            curr['y'] = []
            for d in values:
                ans = d['query'].split('||')[-1]
                curr['y'] += [f'Answer:{ans} (correct)', f'Answer:{ans} (wrong)']
        out.append(curr)
    fname = f"cache/multirc_rag_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_multirc(out, tokenizer, answer_tokenizer, fixed, max_e)
        with open(fname, 'wb') as f:
            pickle.dump((sents, supps, answs, ds, num_s), f)
    return (sents, supps, answs, ds, num_s, sent_labels, labels)

class MultiRCDataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        self.sents, self.supps, self.answs, self.ds, self.num_s, self.sent_labels, self.labels = everything

    def __getitem__(self, idx):
        item = dict()
        item['sents'] = self.sents[idx]
        item['supps'] = self.supps[idx]
        item['answs'] = self.answs[idx]
        item['ds'] = self.ds[idx]
        item['num_s'] = self.num_s[idx]
        item['sent_labels'] = self.sent_labels[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)
