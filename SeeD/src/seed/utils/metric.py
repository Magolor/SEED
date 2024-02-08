from .utils import *

import nltk

class ObjectAccuracy():
    def __init__(self):
        super().__init__()
        
    def compute(predictions, references):
        correct = 0
        for pd, gt in zip(predictions, references):
            if pd == gt:
                correct += 1
        return {"obj_accuracy": correct/len(references)}

class FuzzyStringAccuracy():
    def __init__(self):
        super().__init__()
        
    def compute(predictions, references):
        correct = 0
        for pd, gt in zip(predictions, references):
            pd = re.sub(r'[\W_]', '', pd.lower().strip())
            gt = re.sub(r'[\W_]', '', gt.lower().strip())
            if (not gt) or (pd == gt):
                correct += 1; continue
            if pd and (pd in gt):
                correct += 1; continue
            if gt and (gt in pd):
                correct += 1; continue
        return {"fuzzy_str_accuracy": correct/len(references)}

def get_evaluation_metric(evaluation_metric):
    if evaluation_metric == "accuracy":
        return evaluate.load("accuracy")
    if evaluation_metric == "f1":
        return evaluate.load("f1")
    if evaluation_metric == "obj_accuracy":
        return ObjectAccuracy
    if evaluation_metric == "fuzzy_str_accuracy":
        return FuzzyStringAccuracy