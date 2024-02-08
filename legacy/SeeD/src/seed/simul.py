from .utils import *

def load_model(type, identifier, num_labels=-1):
    config = AutoConfig.from_pretrained(identifier, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(identifier, local_files_only=True)
    if type == 'clslm':
        model = AutoModelForSequenceClassification.from_pretrained(identifier, local_files_only=True,
            num_labels = num_labels,
        )
    elif type == 'seqlm':
        model = AutoModelForSeq2SeqLM.from_pretrained(identifier, local_files_only=True)
    else:
        raise NotImplementedError
    return config, tokenizer, model

class Simul(object):
    def __init__(self, module_path, simul_type, num_classes=-1, sync=256, custom=None):
        self.path = pjoin(module_path, "simul"); CreateFolder(self.path)
        self.type = simul_type; self.num_labels = num_classes
        self.sync = sync; self.c = 0; self.custom = custom; self.clear()

    def finetune(self):
        self.reload()
        dataset = LoadJson(pjoin(self.path, "dataset.jsonl"), backend="jsonl")
        print("Finetuning start...", len(dataset))
        finetune_model(
            type = self.type,
            config = self.config,
            tokenizer = self.tokenizer,
            model = self.model,
            dataset = dataset,
            save_path = self.path,
        )
        print("Finetuning end")
        self.load()

    def reload(self):
        model = pjoin(FROZEN_CKPTS_PATH, self.custom) if self.custom else get_config(f"ckpt-{self.type}"); print(model)
        self.config, self.tokenizer, self.model = load_model(self.type, model, num_labels=self.num_labels); self.finetuned = self.custom is not None

    def load(self):
        self.config, self.tokenizer, self.model = load_model(self.type, self.path, num_labels=self.num_labels); self.finetuned = True

    def save(self):
        self.config.save_pretrained(self.path)
        self.tokenizer.save_pretrained(self.path)
        self.model.save_pretrained(self.path)

    def clear(self):
        self.reload(); self.save()
        SaveJson(list(), pjoin(self.path, "dataset.jsonl"), backend="jsonl")
        ClearFolder(pjoin(self.path, "finetune"), rm=True)
    
    def get_simul(self, key):
        key = key.lower().strip()
        if not self.finetuned:
            return None, None
        if self.type == 'clslm':
            score = classify(key, self.tokenizer, self.model)
            pred = int(score.argmax().item()); confidence = float(2*(score.max().item()-0.5))
            return pred, confidence
    
    def add_simul(self, key, value):
        key = key.lower().strip()
        dataset = LoadJson(pjoin(self.path, "dataset.jsonl"), backend="jsonl")
        if key not in set(d[0] for d in dataset):
            dataset.append((key, value))
            SaveJson(dataset, pjoin(self.path, "dataset.jsonl"), backend="jsonl")
            self.c += 1
        if self.c >= self.sync:
            self.finetune(); self.c = 0

SIMUL_CODE = """
from seed import *

class {name}_simul_class(object):
    def __init__(self, module_path, simul_type, simul_threshold, num_classes, custom=None):
        self.simul = Simul(module_path, simul_type, num_classes, custom=custom)
        self.confidence = simul_threshold
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def {api_self}:
        key = serialize(
            task="{name}",
            {api_copyargs}
        )
        value, confidence = self.simul.get_simul(key)
        print(key, value, confidence, self.confidence)
        if (value is not None) and (confidence >= self.confidence):
            self.responses.append(value)
        else:
            self.responses.append(None)
        return True
    
    def {api_update}:
        key = serialize(
            task="{name}",
            {api_copyargs}
        )
        self.simul.add_simul(key, value)
"""

def format_simul_code(cell):
    return SIMUL_CODE.format(
        name = cell.name,
        api_self = cell.api_self(),
        api_update = cell.api_update(),
        api_copyargs = cell.api_copyargs(),
    )