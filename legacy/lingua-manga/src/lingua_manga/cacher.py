from .utils import *
import faiss

class KVCache(object):
    def __init__(self, path, init=False, sync_updates=32):
        self.path = path
        if init: self.init(model=init)
        else: self.load()
        self.sync_updates = sync_updates; self.sync = 0

    def load(self):
        try:
            self.config = LoadJson(pjoin(self.path, "config.json"))
            self.tokenizer, self.model = load_model(self.config['model'])
            self.index = faiss.read_index(pjoin(self.path, "index.faiss"))
            self.keys = LoadJson(pjoin(self.path, "keys.jsonl"), backend="jsonl")
            self.values = LoadJson(pjoin(self.path, "values.jsonl"), backend="jsonl")
        except:
            self.init(model="sentence-transformers/all-MiniLM-L12-v2")
    
    def save(self):
        SaveJson(self.config, pjoin(self.path, "config.json"), indent=4)
        faiss.write_index(self.index, pjoin(self.path, "index.faiss"))
        SaveJson(self.keys, pjoin(self.path, "keys.jsonl"), backend="jsonl")
        SaveJson(self.values, pjoin(self.path, "values.jsonl"), backend="jsonl")
    
    def init(self, model):
        self.config = {"model": model, "dim": 384}
        self.tokenizer, self.model = load_model(self.config['model'], download=True)
        self.index = faiss.IndexFlatL2(self.config['dim'])
        self.keys = list()
        self.values = list()
        self.save()
    
    def query(self, key):
        if self.index.ntotal == 0:
            return None, None, None
        embed = embedding(key, self.tokenizer, self.model).detach().numpy().reshape(1, -1)
        faiss.normalize_L2(embed)
        distances, indices = self.index.search(embed, 1)
        d, ind = distances[0][0], indices[0][0]
        key, value = self.keys[ind], self.values[ind]
        return key, value, d
        
    def update(self, key, value):
        self.keys.append(key); self.values.append(value)
        embed = embedding(key, self.tokenizer, self.model).detach().numpy().reshape(1, -1)
        faiss.normalize_L2(embed)
        self.index.add(embed); self.sync += 1
        if self.sync >= self.sync_updates:
            self.sync = 0; self.save()