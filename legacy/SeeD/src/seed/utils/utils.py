from pyheaven import *

from typing import *
from dataclasses import dataclass

import io
import contextlib
import IPython.terminal.embed as embed

import os
from os.path import expanduser
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ["TRANSFORMERS_VERBOSITY"] = 'error'
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = 'true'

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data import DefaultDataCollator, DataCollatorForSeq2Seq, DataCollatorWithPadding

import datasets
from datasets import Dataset

import evaluate

import openai

import numpy as np
import pandas as pd

import re
import nltk
import faiss
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzysearch import find_near_matches
from fuzzywuzzy import process

from copy import deepcopy
from collections import Counter

# Config Utils
CONFIG_PATH = pjoin(expanduser("~"), ".seed", "config.json")
def get_config(key=None):
    return LoadJson(CONFIG_PATH)[key] if key else LoadJson(CONFIG_PATH)
def set_config(key, value):
    config = get_config(); config[key] = value; SaveJson(config, CONFIG_PATH, indent=4)
def clr_config():
    CreateFile(CONFIG_PATH); SaveJson(dict(), CONFIG_PATH, indent=4)

# String Utils
import re
def add_indent(code, indent=1, tab=4):
    T = ((" "*tab if tab>=0 else "\t") if isinstance(tab, int) else tab)*indent
    return T + code.strip().replace("\n", "\n"+T)
def parse_code_response(response):
    assert (f"```python" in response), (response)
    assert ("```" in response.split(f"```python")[-1]), (response)
    code = response.split(f"```python")[-1].split("```")[0].strip()
    return code
def parse_code_doc_messages(messages):
    return '"""\n'+('\n\n'.join(["["+message['role']+"]\n"+add_indent(message['content']) for message in messages]))+'\n"""\n'

# Exact Cache Utils
CACHE_PATH = pjoin(expanduser("~"), ".seed", "cache.json")
def serialize(task=None,**kwargs):
    return "|".join(([task] if task else []) + [f"{k}={v}" for k, v in kwargs.items()]).lower().strip()
def get_exact_cache(key):
    key = key.lower().strip(); cache = LoadJson(CACHE_PATH); return cache[key] if (key in cache) else None
def add_exact_cache(key, value):
    key = key.lower().strip(); cache = LoadJson(CACHE_PATH); cache[key] = value; SaveJson(cache, CACHE_PATH, indent=4)
def clr_exact_cache():
    CreateFile(CACHE_PATH); SaveJson(dict(), CACHE_PATH, indent=4)

# Count Utils
def get_llm_count(path="./"):
    return LoadJson(pjoin(path, "llm_count.json"))
def add_llm_count(path="./", type="llm", size=1, tokens=0):
    llm_count = get_llm_count(path)
    assert (type in ['llm', 'cache', 'codegen', 'simul']), "Type must be one of 'llm', 'cache', 'codegen', or 'simul'!"
    if type=='llm':
        llm_count['llm_calls'] += 1
        llm_count['llm_tokens'] += tokens
    llm_count['counts'][type] += size
    SaveJson(llm_count, pjoin(path, "llm_count.json"), indent=4)
def clr_llm_count(path="./"):
    SaveJson({
        "counts": {
            'llm': 0,
            'cache': 0,
            'codegen': 0,
            'simul': 0,
        },
        "llm_calls": 0,
        "llm_tokens": 0,
    }, pjoin(path, "llm_count.json"), indent=4)

# Reorder Utils
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def get_even_clusters(embeds, C):
    n = len(embeds)
    embeds = np.array(embeds)
    kmeans = KMeans(n_clusters=C, random_state=0).fit(embeds)
    centers = kmeans.cluster_centers_.reshape(-1, 1, embeds.shape[-1]).repeat(n//C, 1).reshape(-1, embeds.shape[-1])
    distance_matrix = cdist(embeds, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//(n//C)
    ids_batched = [[j for j in range(n) if clusters[j]==i] for i in range(C)]
    return ids_batched

def find_closest(embeds, embeds_dict):
    d = 1e9; j = -1
    for k, e in embeds_dict.items():
        x = min([np.linalg.norm(embed-e) for embed in embeds])
        if x < d:
            d = x; j = k
    return j

def find_farthest(embeds, embeds_dict):
    d = -1e9; j = -1
    for k, e in embeds_dict.items():
        x = min([np.linalg.norm(embed-e) for embed in embeds])
        if x > d:
            d = x; j = k
    return j

def reorder_data(records, strategy, task="", batch_size=1):
    np.random.seed(42)
    assert (strategy in ['RND', 'SIM', 'DIV', 'CLS', 'FAR']), "Strategy must be one of 'RND', 'SIM', 'DIV', 'CLS', or 'FAR'!"
    if len(records) <= 1:
        return records
    if strategy == 'RND':
        return records
    if strategy == 'CLS':
        embeds_dict = {idx:cache_embed(serialize(task=task,**record)) for idx, record in enumerate(records)}
        idx = np.random.choice(list(embeds_dict.keys()), 1)[0]
        reordered = [(idx, embeds_dict.pop(idx))]
        while len(embeds_dict) > 0:
            # cur_set = reordered[-batch_size:] if batch_size<len(reordered) else reordered
            # idx = find_closest([d[1] for d in cur_set], embeds_dict)
            idx = find_closest([d[1] for d in reordered], embeds_dict)
            reordered.append((idx, embeds_dict.pop(idx)))
        return [records[i] for i, _ in reordered]
    if strategy == 'FAR':
        embeds_dict = {idx:cache_embed(serialize(task=task,**record)) for idx, record in enumerate(records)}
        idx = np.random.choice(list(embeds_dict.keys()), 1)[0]
        reordered = [(idx, embeds_dict.pop(idx))]
        while len(embeds_dict) > 0:
            # cur_set = reordered[-batch_size:] if batch_size<len(reordered) else reordered
            # idx = find_farthest([d[1] for d in cur_set], embeds_dict)
            idx = find_farthest([d[1] for d in reordered], embeds_dict)
            reordered.append((idx, embeds_dict.pop(idx)))
        return [records[i] for i, _ in reordered]

# Experiment Utils
def prepare_data(records, label='label', sample=512, reorder="RND", task="", batch_size=1, seed=42, valid=False):
    np.random.seed(seed)
    np.random.shuffle(records)
    records = records[sample:] if valid else records[:sample]
    records = reorder_data(records, strategy=reorder, task=task, batch_size=batch_size)
    ids = [record['id'] for record in records]
    inputs = [{k:v for k,v in record.items() if k!='id' and k!=label} for record in records]
    labels = [record[label] for record in records]
    return ids, inputs, labels

# Model Utils
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embedding(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    output = mean_pooling(model(**inputs), inputs['attention_mask'])[0]
    return output

def classify(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    return torch.softmax(model(**inputs).logits[0], dim=0)

def inference(x, tokenizer, model):
    inputs = tokenizer([x], max_length=1024, padding=True, truncation=True, return_tensors="pt")
    output_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=0, max_length=512)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

# Vector Cache Utils (TODO: In practice, this part should be persistent, for convenience, we are now loading the model and index every time)
FROZEN_CKPTS_PATH = pjoin(expanduser("~"), ".seed", "ckpts")
try:
    FROZEN_TOKENIZER = AutoTokenizer.from_pretrained(get_config("ckpt-frozen"), local_files_only=True)
    FORZEN_MODEL = AutoModel.from_pretrained(get_config("ckpt-frozen"), local_files_only=True)
except:
    FROZEN_TOKENIZER = AutoTokenizer.from_pretrained("bert-large-cased", local_files_only=True)
    FORZEN_MODEL = AutoModel.from_pretrained("bert-large-cased", local_files_only=True)
def cache_embed(key):
    key = key.lower().strip(); embed = embedding(key, FROZEN_TOKENIZER, FORZEN_MODEL).detach().cpu().numpy().reshape(1, -1); faiss.normalize_L2(embed); return embed

# Simulation Finetuning Utils
def create_dataset(simul_type, dataset):
    if simul_type=='clslm':
        df = pd.DataFrame(dataset, columns=['text', 'label'])
        df['label'] = df['label'].astype(int)
    elif simul_type=='seqlm':
        df = pd.DataFrame(dataset, columns=['source', 'target'])
        df['target'] = df['target'].astype(str)
    else:
        raise NotImplementedError
    return Dataset.from_pandas(df)

def finetune_model(type, config, tokenizer, model, dataset, save_path):
    transformers.set_seed(42)
    if type=='clslm':
        def encode(X):
            return tokenizer(X['text'], truncation=True, max_length=1024)
        encoded_dataset = create_dataset(type, dataset).map(encode, batched=True)
        dataset_split = encoded_dataset.train_test_split(test_size=0.1)
        train_dataset, valid_dataset = dataset_split['train'], dataset_split['test']
        accuracy = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)
        
        training_args = TrainingArguments(
            output_dir=pjoin(save_path, "finetune"),
            num_train_epochs=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            
            optim='adamw_torch',
        )

        trainer = Trainer(
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        raise NotImplementedError
        
    trainer.train()
    model.save_pretrained(save_path)
