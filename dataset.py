from collections import defaultdict, Counter
from glob import glob
from itertools import combinations, product
import json
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

def supp_helper(sentences, title):
    if len(sentences) >= 2:
        s = [sentences[0], sentences[1], sentences[0]+sentences[1]]
    else:
        s = [sentences[0]]
    for i in range(len(s)):
        s[i] = f"{title}: {s[i]}"
    return s

def len_helper(l):
    l.insert(0, 0)
    for i in range(1, len(l)):
        l[i] += l[i-1]
    return l

def preprocess_pipeline_function(examples, tok, answ_tok, max_sent, fixed):
    paragraphs = []
    supps = []
    plengths = []
    slengths = []
    nums = []
    ds = []
    ds2 = []
    num_sents = []
    sent_labels = []
    sent_labels2 = []
    for context, labels, q, supp in zip(examples["context"], examples["labels"], examples["question"], examples['supporting_facts']):
        ts = context["title"]
        sents = context["sentences"]
        # correct paragrpah indices
        x, y = labels
        tmp = defaultdict(list)
        tmp2 = dict()
        for t, sid in zip(supp["title"], supp["sent_id"]):
            if t == ts[x]:
                if sid >= len(sents[x]):
                    print("INDEX OUT OF RANGE", sid, len(sents[x]))
                    continue
                tmp[x].append(sid)
            else:
                if sid >= len(sents[y]):
                    print("INDEX OUT OF RANGE", sid, len(sents[y]))
                    continue
                tmp[y].append(sid)
        sent_labels.append(tmp)
        ps = []
        idx2p = dict()
        p2idx = defaultdict(list)
        cnt = 0
        curr_sents = []
        for i in range(len(ts)):
            curr_sents.append(len(sents[i][:max_sent]))
            for j in range(0, len(sents[i][:max_sent]), fixed):
                ps.append(sents[i][j:j+fixed])
                idx2p[cnt] = i
                p2idx[i].append(cnt)
                cnt += 1
        num_sents.append(sum(curr_sents))
        curr_ps = [f' {tok.unk_token}'.join(p) for p in ps]
        paragraphs += [f"{q} {tok.unk_token} {p}" for p in curr_ps]
        plengths.append(len(ps))
        ds.append(idx2p)
        ds2.append(p2idx)
        curr_supps = []
        for i in range(len(ts)):
            rang = range(len(sents[i][:max_sent]))
            combs = list(combinations(rang, r=1)) + list(combinations(rang, r=2))
            tmp2[i] = combs
            for c in combs:
                curr_sents = [sents[i][j] for j in c]
                curr_sents = ' '.join(curr_sents)
                curr_sents = f"{q} {answ_tok.sep_token} {curr_sents}"
                curr_supps.append(curr_sents)
            slengths.append(len(combs))
        supps += curr_supps
        nums.append(len(ts))
        sent_labels2.append(tmp2)
    plengths = len_helper(plengths)
    slengths = len_helper(slengths)
    nums = len_helper(nums)
    tokenized_paras = tok(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_paras = [tokenized_paras[plengths[i]:plengths[i+1]] for i in range(len(plengths)-1)]
    answers = [a for a in examples['answer']]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    tokenized_supps = [tokenized_supps[nums[i]:nums[i+1]] for i in range(len(nums)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_paras)
    return tokenized_paras, tokenized_supps, tokenized_answers, (ds, ds2, num_sents, sent_labels, sent_labels2)

def prepare_pipeline(tokenizer, answ_tokenizer, split, data, max_sent, k=1, fixed=False, sentence=False, baseline=False):
    assert fixed > 0
    print("preparing pipeline")
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
    fname = f"cache/unsupervised_hotpotqa_encodings_{k}_{fixed}_{split}.pkl"

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            paras, supps, answers, ds = pickle.load(f)
    else:
        paras, supps, answers, ds = preprocess_pipeline_function(data, tokenizer, answ_tokenizer, max_sent, fixed)
        with open(fname, 'wb') as f:
            pickle.dump((paras, supps, answers, ds), f)
    return paras, supps, answers, ds

def preprocess_simplified_function(examples, tok, answ_tok, max_sent, fixed):
    paragraphs = []
    supps = []
    plengths = []
    ds = []
    for context, labels, q, supp in zip(examples["context"], examples["labels"], examples["question"], examples['supporting_facts']):
        ts = context["title"]
        sents = context["sentences"]
        # correct paragrpah indices
        x, y = labels[:2]
        tmp = defaultdict(list)
        for t, sid in zip(supp["title"], supp["sent_id"]):
            if t == ts[x]:
                if sid >= len(sents[x]):
                    print("INDEX OUT OF RANGE", sid, len(sents[x]))
                    continue
                tmp[x].append(sid)
            else:
                if sid >= len(sents[y]):
                    print("INDEX OUT OF RANGE", sid, len(sents[y]))
                    continue
                tmp[y].append(sid)
        ps = []
        idx2p = dict()
        cnt = 0
        for i in range(len(labels)):
            for j in range(0, len(sents[labels[i]][:max_sent]), fixed):
                ps.append(sents[labels[i]][j:j+fixed])
                idx2p[cnt] = i
                cnt += 1
        plengths.append(len(ps))
        ds.append(idx2p)
        ps = [' '.join(p) for p in ps]
        paragraphs += [f"{q} {tok.sep_token} {p}" for p in ps]
        for i in range(plengths[-1]):
            ps[i] = f'{ts[labels[idx2p[i]]]}: {ps[i]}'
        for m in ps:
            p = f"{q} {answ_tok.sep_token} {m}"
            supps.append(p)
    #torch.save((examples["question"], paragraphs, examples['answer'], examples["labels"]), "data.pt")
    plengths = len_helper(plengths)
    tokenized_paras = tok(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_paras = [tokenized_paras[plengths[i]:plengths[i+1]] for i in range(len(plengths)-1)]
    answers = [a for a in examples['answer']]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[plengths[i]:plengths[i+1]] for i in range(len(plengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_paras)
    return tokenized_paras, tokenized_supps, tokenized_answers, ds

def prepare_simplified(tokenizer, answ_tokenizer, split, data, max_sent, k=1, fixed=False, sentence=False, baseline=False):
    assert fixed > 0
    print("preparing simplified")
    data = data[split][:]

    remained = []
    for title_label, titles in zip(data['supporting_facts'], data['context']):
        title_label = title_label["title"]
        titles = titles["title"]
        l = []
        for i in range(len(titles)):
            if titles[i] in title_label:
                l.append(i)
        assert len(l) == 2
        assert l[0] < l[1]
        curr_k = len(titles) - 2
        total = list(range(len(titles)))
        total.remove(l[0])
        total.remove(l[1])
        distractor = random.sample(total, k=curr_k)
        l += distractor
        remained.append(l)
    data["labels"] = remained
    fname = f"cache/partition_hotpotqa_simplified_encodings_{k}_{fixed}_{split}.pkl"

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            paras, supps, answers, ds = pickle.load(f)
    else:
        paras, supps, answers, ds = preprocess_simplified_function(data, tokenizer, answ_tokenizer, max_sent, fixed)
        with open(fname, 'wb') as f:
            pickle.dump((paras, supps, answers, ds), f)
    return paras, supps, answers, ds

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

class SimplifiedHotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, paras, supps, answs, ds):
        self.paras = paras
        self.supps = supps
        self.answs = answs
        self.ds = ds

    def __getitem__(self, idx):
        item = dict()
        item["paras"] = self.paras[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["ds"] = self.ds[idx]
        return item

    def __len__(self):
        return len(self.answs)

class UnsupHotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, paras, supps, answs, ds):
        self.paras = paras
        self.supps = supps
        self.answs = answs
        self.ds, self.ds2, self.num_sents, self.slabel, self.slabels2 = ds

    def __getitem__(self, idx):
        item = dict()
        item = dict()
        item["paras"] = self.paras[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["ds"] = self.ds[idx]
        item["ds2"] = self.ds2[idx]
        item["num_s"] = self.num_sents[idx]
        item["s_labels"] = self.slabel[idx]
        item["s_maps"] = self.slabels2[idx]
        return item

    def __len__(self):
        return len(self.answs)

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
        for i in range(1, min(max_e+1, z_len+1)):
            curr_curr_idxes = []
            for j in range(z_len-i+1):
                curr_curr_idxes.append(rang[j:j+i])
                curr_supps = [e['z'][m] for m in curr_curr_idxes[-1]]
                curr_supps = ' '.join(curr_supps)
                curr_supps = e['x'] + f' {answ_tok.sep_token} ' + curr_supps
                supps.append(curr_supps)
            curr_idxes.append(curr_curr_idxes)
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
    fname = f"cache/fever_{split}.pkl"
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

def prepare_multirc(tokenizer, answer_tokenizer, split, docs, fixed, max_e, path="data/multirc/"):
    print(f"prepare MultiRC {split}")
    data = []
    with open(f"{path}/{split}.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    out = []
    labels = []
    sent_labels = []
    counts_z = []
    counts_gold = []
    for d in data:
        curr = dict()
        curr['x'] = d['query']
        d['evidences'] = [ee for e in d['evidences'] for ee in e]
        docid = [l['docid'] for l in d['evidences']]
        docid = set(docid)
        assert len(docid) == 1
        docid = docid.pop()
        curr['z'] = docs[docid]
        counts_z.append(len(curr['z']))
        gold_z = [l['start_sentence'] for l in d['evidences']]
        counts_gold.append(len(gold_z))
        sent_labels.append(gold_z)
        curr['y'] = d['classification'].lower()
        label = 0 if curr['y'] == "supports" else 1
        labels.append(label)
        out.append(curr)
    print(Counter(counts_z))
    print(Counter(counts_gold))
    fname = f"cache/fever_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_fever(out, tokenizer, answer_tokenizer, fixed, max_e)
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
