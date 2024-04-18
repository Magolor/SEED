from pyheaven import *

import os
import sys
import pkg_resources
from os.path import expanduser

from typing import *
from dataclasses import dataclass

import re
import textwrap

from itertools import combinations
from functools import partial
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import AutoModelForTextEncoding, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback

import datasets
from datasets import Dataset

import evaluate

import ray.tune as tune

import io
import contextlib
import IPython.terminal.embed as embed

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

import numpy as np
import pandas as pd

import faiss

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = 'true'

def merge_dicts(l):
    merged = {}
    for d in l:
        for key, value in d.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_dicts([merged[key], value])
            else:
                merged[key] = value
    return merged

def LoadText(path):
    with open(path, 'r') as f:
        return f.read()

RESOURCES_PATH = pjoin("resources")
def find_resource(name):
    return pkg_resources.resource_filename('seed', pjoin(RESOURCES_PATH, name))

CONFIG_PATH = pjoin(expanduser("~"), ".seed2", "config.json")
def init_config(**kwargs):
    CreateFile(CONFIG_PATH); SaveJson(merge_dicts([LoadJson(find_resource('configs/default.json')), kwargs]), CONFIG_PATH, indent=4)
def get_config(key=None):
    config = LoadJson(CONFIG_PATH); return config if key is None else (config[key] if key in config else None)
def set_config(key, value):
    config = get_config(); config[key] = value; SaveJson(config, CONFIG_PATH, indent=4)

CACHE_PATH = pjoin(expanduser("~"), ".seed2", "cache.json")
def get_exact_cache(key):
    key = key.lower().strip(); cache = LoadJson(CACHE_PATH); return cache[key] if (key in cache) else None
def add_exact_cache(key, value):
    key = key.lower().strip(); cache = LoadJson(CACHE_PATH)
    try:
        SaveJson(cache | {key: value}, CACHE_PATH, indent=4)
    except Exception as e:
        print(e)
def init_exact_cache(force=False):
    CreateFile(CACHE_PATH)
    if force or (not os.path.exists(CACHE_PATH)) or (os.path.getsize(CACHE_PATH) <= 0):
        SaveJson(dict(), CACHE_PATH, indent=4)

def safe_eval(s, globals={}, locals={}):
    try:
        return eval(s, globals, locals)
    except Exception as e:
        return s
def indent(s, tab=1):
    return textwrap.indent(s, (tab*"\t" if isinstance(tab, int) else tab))
def comment(s):
    return '# ' + s.strip().replace('\n', '\n# ')
    # return textwrap.indent(s, '# ')

CKPTS_PATH = pjoin(expanduser("~"), ".seed2", "ckpts")
TEMP_TRAINING_PATH = pjoin(expanduser("~"), ".seed2", "temp")
def init_ckpts():
    CreateFolder(CKPTS_PATH)
def add_ckpt(ckpt):
    C = AutoConfig.from_pretrained(ckpt); C.save_pretrained(pjoin(CKPTS_PATH, ckpt))
    T = AutoTokenizer.from_pretrained(ckpt); T.save_pretrained(pjoin(CKPTS_PATH, ckpt))
    M = AutoModel.from_pretrained(ckpt); M.save_pretrained(pjoin(CKPTS_PATH, ckpt))
def load_ckpt(ckpt, model_type=AutoModel, **kwargs):
    C = AutoConfig.from_pretrained(ckpt, local_files_only=True)
    T = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    M = model_type.from_pretrained(ckpt, local_files_only=True, **kwargs)
    return C, T, M
def load_model(ckpt, model_type=AutoModel, **kwargs):
    if ExistFolder(ckpt):
        return load_ckpt(ckpt, model_type=model_type, **kwargs)
    if ExistFolder(pjoin(CKPTS_PATH, ckpt)):
        return load_ckpt(pjoin(CKPTS_PATH, ckpt), model_type=model_type, **kwargs)
    add_ckpt(ckpt); return load_ckpt(pjoin(CKPTS_PATH, ckpt), model_type=model_type, **kwargs)
def save_model(ckpt, C, T, M, profile=None):
    CreateFolder(ckpt)
    C.save_pretrained(ckpt)
    T.save_pretrained(ckpt)
    M.save_pretrained(ckpt)
    if profile is not None:
        SaveJson(profile, pjoin(ckpt, "profile.json"), indent=4)
def save_ckpt(ckpt, C, T, M, profile=None):
    save_model(pjoin(CKPTS_PATH, ckpt), C, T, M, profile=profile)

# def serialize(task=None,**kwargs):
#     return "|".join(([task] if task else []) + [f"{k}={v}" for k, v in kwargs.items()]).lower().strip()
def serialize(name, inputs, outputs=None):
    return f"assert {name}({', '.join([k+'='+repr(v) for k,v in sorted(inputs.items())])}) == " + (f"({', '.join([repr(v) for k,v in sorted(outputs.items())])}))" if outputs else "")

def partition_list(L, r=0.8):
    np.random.shuffle(L); return L[:int(len(L)*r)], L[int(len(L)*r):]

def ignore_label(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower().strip()=='nan':
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False