from collections import defaultdict
from itertools import combinations, product
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

def preprocess_simplified_sent_function(examples, tok, answ_tok, max_sent, fixed):
    supps = []
    lengths = []
    found = 0
    assert tok.sep_token == answ_tok.sep_token
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
        #not_supps0 = [sents[x][i] for i in range(len(sents[x])) if i not in tmp[x]]
        #not_supps1 = [sents[y][i] for i in range(len(sents[y])) if i not in tmp[y]]
        #gold0 = [sents[x][i] for i in tmp[x]]
        #gold1 = [sents[y][i] for i in tmp[y]]
        #if len(not_supps0) != 0 or len(not_supp1) != 0:
        #    distractor0 = 0
        if tmp[x] in [[0], [1], [0, 1]] and tmp[y] in [[0], [1], [0, 1]]:
            found += 1
        p1 = supp_helper(sents[x], ts[x])
        p2 = supp_helper(sents[y], ts[y])
        curr_supps = []
        for m, n in product(p1, p2):
            curr_supps.append(m+f" {tok.sep_token} "+n)
        supps += curr_supps
        lengths.append(len(curr_supps))
    print(found / len(examples["question"]))
    para_length = len(labels)
    supp_length = len(list(combinations(labels, 2)))
    tokenized_paras = tok(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_paras = [tokenized_paras[i:i+para_length] for i in range(0, len(tokenized_paras), para_length)]
    answers = [a for a in examples['answer']]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[i:i+supp_length] for i in range(0, len(tokenized_supps), supp_length)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_paras)
    return tokenized_paras, tokenized_supps, tokenized_answers

def get_masks(original, masked):
    masks = []
    for x, y in zip(original, masked):
        mask = [1] * len(x)
        for i in range(len(mask)):
            if x[i] not in y:
                mask[i] = 0
        masks.append(mask)
    return masks

def preprocess_simplified_function(examples, tok, answ_tok, max_sent, fixed, entities):
    paragraphs = []
    supps = []
    masked_paragraphs = []
    masked_supps = []
    for context, labels, q, supp, ents in zip(examples["context"], examples["labels"], examples["question"], examples['supporting_facts'], entities):
        ts = context["title"]
        sents = context["sentences"]
        # correct paragrpah indices
        x, y = labels[:2]
        if len(ts) != 10:
            ts = correct_format(ts)
            sents = correct_format(sents)
        tmp = defaultdict(list)
        ents = [ee for ent in ents for e in ent for ee in e]
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
        if fixed:
            ps = []
            for l in labels:
                ps.append(sents[l][:fixed])
        else:
            ps = [[sents[x][i] for i in tmp[x]], [sents[y][i] for i in tmp[y]]]
            len0 = len(ps[0])
            len1 = len(ps[1])
            for i in labels[2:]:
                rand = random.random()
                if rand >= 0.5:
                    ps.append(sents[i][:len0])
                else:
                    ps.append(sents[i][:len1])
        ps = [' '.join(p) for p in ps]
        q_ps = [f"{q} {tok.sep_token} {p}" for p in ps]
        paragraphs += q_ps
        curr_masked_ps = []
        for i in range(len(q_ps)):
            modified = q_ps[i]
            for e in ents:
                modified = modified.replace(e, tok.mask_token)
            curr_masked_ps.append(modified)
        masked_paragraphs += curr_masked_ps
        for (m, n) in combinations(ps, 2):
            p = f"{q} {answ_tok.sep_token} {m} {answ_tok.sep_token} {n}"
            supps.append(p)
        for (m, n) in combinations(curr_masked_ps, 2):
            p = f"{q} {answ_tok.sep_token} {m} {answ_tok.sep_token} {n}"
            masked_supps.append(p)
    para_length = len(labels)
    supp_length = len(list(combinations(labels, 2)))
    tokenized_paras = tok(paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    masked_paras = tok(masked_paragraphs, truncation=True, return_attention_mask=False)['input_ids']
    pmasks = get_masks(tokenized_paras, masked_paras)
    tokenized_paras = [tokenized_paras[i:i+para_length] for i in range(0, len(tokenized_paras), para_length)]
    pmasks = [pmasks[i:i+para_length] for i in range(0, len(pmasks), para_length)]
    answers = [a for a in examples['answer']]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    masked_supps = answ_tok(masked_supps, truncation=True, return_attention_mask=False)['input_ids']
    smasks = get_masks(tokenized_supps, masked_supps)
    tokenized_supps = [tokenized_supps[i:i+supp_length] for i in range(0, len(tokenized_supps), supp_length)]
    smasks = [smasks[i:i+supp_length] for i in range(0, len(smasks), supp_length)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_paras)
    return tokenized_paras, tokenized_supps, tokenized_answers, pmasks, smasks

def prepare_simplified(tokenizer, answ_tokenizer, split, data, max_sent, k=1, fixed=False, sentence=False, baseline=False):
    print("preparing simplified")
    data = data[split][:]
    entities = torch.load(f"cache/hotpotqa_{split}.pt")

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
        total = list(range(10))
        total.remove(l[0])
        total.remove(l[1])
        distractor = random.sample(total, k=k)
        l += distractor
        remained.append(l)
    data["labels"] = remained
    if sentence:
        fname = f"cache/hotpotqa_entities_simplified_sent_encodings.pkl"
        proc_function = preprocess_simplified_sent_function
    else:
        fname = f"cache/hotpotqa_entities_simplified_encodings_{k}.pkl"
        if fixed > 0:
            fname = fname.replace(".pkl", f"_fixed{fixed}.pkl")
        proc_function = preprocess_simplified_function

    if split == "train":
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                paras, supps, answers, pmasks, smasks = pickle.load(f)
        else:
            paras, supps, answers, pmasks, smasks = proc_function(data, tokenizer, answ_tokenizer, max_sent, fixed, entities)
            with open(fname, 'wb') as f:
                pickle.dump((paras, supps, answers, pmasks, smasks), f)
    else:
        paras, supps, answers, pmasks, smasks = proc_function(data, tokenizer, answ_tokenizer, max_sent, fixed, entities)

    if split == "train":
        paras, supps, pmasks, smasks, answers = split_data(paras, supps, pmasks, smasks, answers)
    return paras, supps, answers, pmasks, smasks

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

class SimplifiedHotpotQADataset(torch.utils.data.Dataset):
    def __init__(self, paras, supps, answs, pmasks, smasks):
        self.paras = paras
        self.supps = supps
        self.answs = answs
        self.pmasks = pmasks
        self.smasks = smasks

    def __getitem__(self, idx):
        item = dict()
        item["paras"] = self.paras[idx]
        item["supps"] = self.supps[idx]
        item["answs"] = self.answs[idx]
        item["pmasks"] = self.pmasks[idx]
        item["smasks"] = self.smasks[idx]
        return item

    def __len__(self):
        return len(self.answs)

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
