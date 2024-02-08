from .utils import *

class Cache(object):
    def __init__(self, module_path, sync=32):
        self.path = pjoin(module_path, "cache"); CreateFolder(self.path)
        self.sync = sync; self.c = 0; self.clear()

    def load(self):
        self.index = faiss.read_index(pjoin(self.path, "index.faiss"))
        self.keys = LoadJson(pjoin(self.path, "keys.jsonl"), backend="jsonl")
        self.values = LoadJson(pjoin(self.path, "values.jsonl"), backend="jsonl")

    def save(self):
        faiss.write_index(self.index, pjoin(self.path, "index.faiss"))
        SaveJson(self.keys, pjoin(self.path, "keys.jsonl"), backend="jsonl")
        SaveJson(self.values, pjoin(self.path, "values.jsonl"), backend="jsonl")

    def clear(self):
        self.index, self.keys, self.values = faiss.IndexFlatL2(get_config("cache-dim")), list(), list(); self.save()
    
    def get_cache(self, key):
        key = key.lower().strip()
        embed = cache_embed(key)
        if self.index.ntotal == 0:
            return None, None, None
        distances, indices = self.index.search(embed, 1)
        dist, ind = distances[0][0], indices[0][0]
        key, value = self.keys[ind], self.values[ind]
        return key, value, dist
    
    def add_cache(self, key, value):
        key = key.lower().strip()
        if key not in self.keys:
            self.keys.append(key); self.values.append(value)
            embed = cache_embed(key)
            self.index.add(embed)
            self.c += 1
        if self.c >= self.sync:
            self.save(); self.c = 0

CACHE_CODE = """
from seed import *

class {name}_cache_class(object):
    def __init__(self, module_path, cache_threshold):
        self.threshold = cache_threshold
        self.cache = Cache(module_path)
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def {api_self}:
        key = serialize(
            task="{name}",
            {api_copyargs}
        )
        found, value, dist = self.cache.get_cache(key)
        if (found is not None) and (dist <= 2*self.threshold):
            self.responses.append(value)
        else:
            self.responses.append(None)
        return True
    
    def {api_update}:
        key = serialize(
            task="{name}",
            {api_copyargs}
        )
        self.cache.add_cache(key, value)
"""

def format_cache_code(cell):
    return CACHE_CODE.format(
        name = cell.name,
        api_self = cell.api_self(),
        api_update = cell.api_update(),
        api_copyargs = cell.api_copyargs(),
    )