from pyheaven import *

from typing import *
from dataclasses import dataclass

import os
from os.path import expanduser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import contextlib
import IPython.terminal.embed as embed


import openai

import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data import DefaultDataCollator, DataCollatorForSeq2Seq

import datasets
from datasets import Dataset

# lingua manga utils
LG_ROOT = pjoin(expanduser("~"), ".lingua-manga")
LG_CKPTS = pjoin(LG_ROOT, "ckpts")

def GetConfig(config):
    return LoadJson(pjoin(LG_ROOT, f"{config}.json"))

# string utils
import re
def add_indent(code, indent=1, tab=4):
    T = ((" "*tab if tab>=0 else "\t") if isinstance(tab, int) else tab)*indent
    return T + code.strip().replace("\n", "\n"+T)

def find_command(text):
    return re.findall(r'<[^<>]+>', text)

def parameterize(key, param="data"):
    return f"{param}['{key}']"

def parse_keys(data):
    return "\n".join([f"{k}: {v}" for k,v in data.items()]) if len(data)>0 else "None"

def parse_inputs(inputs):
    return f"Input:\n" + add_indent("\n".join([f"{k}={repr(v)}" for k,v in inputs.items()]) if len(inputs)>0 else "None")

def parse_outputs(outputs):
    return f"Output:\n" + add_indent("\n".join([f"{k}={repr(v)}" for k,v in outputs.items()]) if len(outputs)>0 else "None")

def parse_example(inputs, outputs, i=None, info=''):
    return "\n".join([("Example:" if i is None else f"Example #{i}:"),add_indent(parse_inputs(inputs)),add_indent(parse_outputs(outputs))] + ([add_indent(f"Explanation:\n"+add_indent(info))] if info else []))

def parse_examples(examples):
    return "\n".join([parse_example(inputs, outputs, i) for i, (inputs, outputs) in enumerate(examples)]) if len(examples)>0 else "None"

def parse_pro_examples(examples):
    return "\n".join([parse_example(example['inputs'], example['outputs'], i, info=example['info']) for i, example in enumerate(examples)]) if len(examples)>0 else "None"

def parse_instance(inputs, i=None):
    return "\n".join([("Instance:" if i is None else f"Instance #{i}:"),add_indent(parse_inputs(inputs))])

def parse_instances(examples):
    return "\n".join([parse_instance(inputs, i) for i, (inputs, outputs) in enumerate(examples)]) if len(examples)>0 else "None"

def parse_interaction_examples(examples):
    return "\n".join([("Example:" if i is None else f"Example #{i}:")+"\n"+add_indent(example) for i, example in enumerate(examples)]) if len(examples)>0 else "None"

# cache utils
def retrieve_cache(key):
    key = key.lower().strip()
    cache = LoadJson(pjoin(LG_ROOT, "cache.json"))
    if key in cache:
        return cache[key]
    return None

def update_cache(key, value, update=False):
    key = key.lower().strip()
    cache = LoadJson(pjoin(LG_ROOT, "cache.json"))
    if update or (key not in cache):
        cache[key] = value
    SaveJson(cache, pjoin(LG_ROOT, "cache.json"))

def clear_cache():
    SaveJson(dict(), pjoin(LG_ROOT, "cache.json"))

# model utils
def load_model(model_name, download=False, model_type=AutoModel, *args, **kwargs):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = model_type.from_pretrained(model_name, local_files_only=True)
    except:
        path = pjoin(LG_CKPTS, model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = model_type.from_pretrained(path, local_files_only=True, *args, **kwargs)
        except:
            if download:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = model_type.from_pretrained(model_name)
                tokenizer.save_pretrained(path)
                model.save_pretrained(path)
                model = model_type.from_pretrained(path, local_files_only=True, *args, **kwargs)
            else:
                tokenizer, model = None, None
    return tokenizer, model

def create_dataset(dataset):
    return Dataset.from_dict({key: [dic[key] for dic in dataset] for key in dataset[0]})

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def classify(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    return torch.softmax(model(**inputs).logits[0], dim=0)

def embedding(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    output = mean_pooling(model(**inputs), inputs['attention_mask'])[0]
    return output

def inference(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    output_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=0, max_length=512)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

# exp utils
class EfficiencyCounter(object):
    def __init__(self):
        self.c = Counter()
        
    def clear(self):
        self.c = Counter()
        
    def add(self, key):
        self.c.update([key, "total"])
    
    def profile(self):
        return {k: (v/self.c["total"] if k!='total' else v) for k, v in self.c.items()}
counter = EfficiencyCounter()