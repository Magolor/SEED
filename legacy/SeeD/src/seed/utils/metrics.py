from .utils import *

def evaluate_llm_count(output_file):
    llm_count = get_llm_count()
    with open(output_file, "a") as f:
        total = sum(llm_count['counts'][k] for k in llm_count['counts'])
        f.write(f"Total      = {total:4d}\n")
        f.write(f"Cache      = {llm_count['counts']['cache']:4d};\tRatio = {llm_count['counts']['cache']/total*100 if total else 0.:8.4f}%\n")
        f.write(f"Code Gen   = {llm_count['counts']['codegen']:4d};\tRatio = {llm_count['counts']['codegen']/total*100 if total else 0.:8.4f}%\n")
        f.write(f"Simul      = {llm_count['counts']['simul']:4d};\tRatio = {llm_count['counts']['simul']/total*100 if total else 0.:8.4f}%\n")
        f.write(f"LLM Query  = {llm_count['counts']['llm']:4d};\tRatio = {llm_count['counts']['llm']/total*100 if total else 0.:8.4f}%\n")
        f.write(f"LLM #Toks  = {llm_count['llm_tokens']:8d}\n")
        f.write(f"LLM #Calls = {llm_count['llm_calls']:4d}\n")

def evaluate_binary(output_file, pds, gts):
    TP, FP, TN, FN = 0, 0, 0, 0
    for pd, gt in zip(pds, gts):
        TP += gt and pd
        FP += (not gt) and pd
        TN += (not gt) and (not pd)
        FN += gt and (not pd)
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    with open(output_file, "a") as f:
        f.write("Pre", precision_score(gts, pds), "\n")
        f.write("Rec", recall_score(gts, pds), "\n")
        f.write("F1", f1_score(gts, pds), "\n")
        f.write("Acc", accuracy_score(gts, pds), "\n")

def evaluate_list(output_file, pds, gts):
    P, G, C = [], [], []
    for pd, gt in zip(pds, gts):
        p = [(x if isinstance(x, str) else x[0]).lower().replace('-','_') for x in pd if x] if pd else []
        g = [(x if isinstance(x, str) else x[0]).lower().replace('-','_') for x in gt]
        c = [x for x in g if x in p]
        P.append(len(p)); G.append(len(g)); C.append(len(c))
    mPre = sum(C)/sum(P) if sum(P) else 0.
    mRec = sum(C)/sum(G) if sum(G) else 0.
    mF1 = mPre*mRec*2/(mPre+mRec) if mPre+mRec else 0.
    MPres = [(c/p if p else 1) for c,p,g in zip(C,P,G)]
    MRecs = [(c/g if g else 1) for c,p,g in zip(C,P,G)]
    MPre = sum(MPres)/len(C)
    MRec = sum(MRecs)/len(C)
    MF1 = sum([2*p*r/(p+r) if p+r else 0. for p,r in zip(MPres,MRecs)])/len(C)
    with open(output_file, "a") as f:
        f.write(f"mPre  = {mPre*100:8.4f}%\n")
        f.write(f"mRec  = {mRec*100:8.4f}%\n")
        f.write(f"mF1   = {mF1*100:8.4f}%\n")
        f.write(f"MPre  = {MPre*100:8.4f}%\n")
        f.write(f"MRec  = {MRec*100:8.4f}%\n")
        f.write(f"MF1   = {MF1*100:8.4f}%\n")

def evaluate_fuzzy(output_file, pds, gts):
    T, F = 0, 0
    for pd, gt in zip(pds, gts):
        pd = re.sub(r'[\W_]', '', pd.lower().strip())
        gt = re.sub(r'[\W_]', '', gt.lower().strip())
        if (not gt) or (pd == gt):
            T += 1; continue
        if pd and (pd in gt):
            T += 1; continue
        if gt and (gt in pd):
            T += 1; continue
        F += 1
    with open(output_file, "a") as f:
        f.write(f"T F = {T:4d} {F:4d}\n")
        f.write(f"Acc = {T/(T+F)*100:8.4f}%\n")


class TokenF1Metric(object):
    WHITESPACE_AND_PUNCTUATION = set([' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'])
    ARTICLES = set(['the', 'a', 'an'])
    
    @staticmethod
    def CleanAnswer(answer):
        answer = answer.lower()
        answer = answer.replace(u'\u00a0', ' ')
        while len(answer) > 1 and answer[0] in TokenF1Metric.WHITESPACE_AND_PUNCTUATION:
            answer = answer[1:]
        while len(answer) > 1 and answer[-1] in TokenF1Metric.WHITESPACE_AND_PUNCTUATION:
            answer = answer[:-1]

        answer = answer.split()
        if len(answer) > 1 and answer[0] in TokenF1Metric.ARTICLES:
            answer = answer[1:]
        answer = ' '.join(answer)

        return answer

    @staticmethod
    def F1Single(pd, gt):
        def GetTokens(text):
            text = TokenF1Metric.CleanAnswer(text)
            for delimeter in TokenF1Metric.WHITESPACE_AND_PUNCTUATION:
                text = text.replace(delimeter, ' ')
            return text.split()
        pd_tokens = Counter(GetTokens(pd)); n_pd_tokens = sum(pd_tokens.values())
        gt_tokens = Counter(GetTokens(gt)); n_gt_tokens = sum(gt_tokens.values())
        n_same = sum((pd_tokens & gt_tokens).values())
        if n_gt_tokens==0 and n_pd_tokens==0:
            return 1., 1., 1.
        if n_gt_tokens==0:
            return 0., 1., 0.
        if n_pd_tokens==0:
            return 1., 0., 0.
        pre = n_same / n_pd_tokens; rec = n_same / n_gt_tokens
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.
        return pre, rec, f1

def evaluate_token_f1(output_file, pds, gts, thres=75):
    mPre, mRec, mF1 = 0., 0., 0.
    for pd, gt in zip(pds, gts):
        pre, rec, f1 = TokenF1Metric.F1Single(pd, gt)
        mPre += pre; mRec += rec; mF1 += f1
    with open(output_file, "a") as f:
        f.write(f"mPre = {(mPre/len(gts))*100:8.4f}%\n")
        f.write(f"mRec = {(mRec/len(gts))*100:8.4f}%\n")
        f.write(f"mF1  = {(mF1/len(gts))*100:8.4f}%\n")