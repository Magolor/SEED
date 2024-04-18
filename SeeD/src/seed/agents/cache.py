from ..templates import *
from .agent import Agent

class CacheAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = pjoin(self.config['project_path'], "agents", "cache")
        self.model = load_model(self.config['cache_frozen_ckpt'], model_type=AutoModelForTextEncoding)
        self.C = self.model[0]; self.T = self.model[1]; self.M = self.model[2]
        self.load()
    
    def load(self, init=False):
        self.info = LoadJson(pjoin(self.path, "info.json"))                             if (not init) and ExistFile(pjoin(self.path, "info.json"))         else dict({'last_sync': 0})
        self.S = set(LoadJson(pjoin(self.path, "identifiers.jsonl"), backend="jsonl"))  if (not init) and ExistFile(pjoin(self.path, "identifiers.jsonl")) else set()
        self.K = LoadJson(pjoin(self.path, "keys.jsonl"), backend="jsonl")              if (not init) and ExistFile(pjoin(self.path, "keys.jsonl"))        else list()
        self.V = LoadJson(pjoin(self.path, "values.jsonl"), backend="jsonl")            if (not init) and ExistFile(pjoin(self.path, "values.jsonl"))      else list()
        self.D = LoadJson(pjoin(self.path, "data_split.jsonl"), backend="jsonl")        if (not init) and ExistFile(pjoin(self.path, "data_split.jsonl"))  else list()
        self.B = set(LoadJson(pjoin(self.path, "indexed.jsonl"), backend="jsonl"))      if (not init) and ExistFile(pjoin(self.path, "indexed.jsonl"))     else set()
        self.I = faiss.read_index(pjoin(self.path, "index.faiss"))                      if (not init) and ExistFile(pjoin(self.path, "index.faiss"))       else faiss.IndexIDMap(faiss.IndexFlatIP(self.C.hidden_size))
        self.conf = LoadJson(pjoin(self.path, "confidences.jsonl"), backend="jsonl")    if (not init) and ExistFile(pjoin(self.path, "confidences.jsonl")) else self.get_confidence_distribution()
    
    def save(self):
        SaveJson(self.info, pjoin(self.path, "info.json"))
        SaveJson(sorted(list(self.S)), pjoin(self.path, "identifiers.jsonl"), backend="jsonl")
        SaveJson(self.K, pjoin(self.path, "keys.jsonl"), backend="jsonl")
        SaveJson(self.V, pjoin(self.path, "values.jsonl"), backend="jsonl")
        SaveJson(self.D, pjoin(self.path, "data_split.jsonl"), backend="jsonl")
        SaveJson(sorted(list(self.B)), pjoin(self.path, "indexed.jsonl"), backend="jsonl")
        faiss.write_index(self.I, pjoin(self.path, "index.faiss"))
        SaveJson(self.conf, pjoin(self.path, "confidences.jsonl"), backend="jsonl")
    
    def compile(self):
        CreateFolder(self.path); CreateFile(pjoin(self.path,'__init__.py'))
        with open(pjoin(self.path,'__init__.py'), 'w') as f:
            f.write(format_cache_code(**self.config))
        self.load(init=True); self.save()
    
    def update(self, instance, label, **kwargs):
        if ignore_label(label): label="nan"
        s = serialize(self.config['name'], instance)
        e = norm_embed(s, self.C, self.T, self.M)
        if s in self.S: return
        
        self.S.add(s)
        self.K.append(instance)
        self.V.append(label)
        self.D.append('dev' if ignore_label(label) else 'train')
        if self.D[-1]=='train':
            self.I.add_with_ids(e, len(self.S)-1)
            self.B.add(len(self.S)-1)
        if self.I.ntotal > self.config['cache_max_size']:
            idx = np.random.choice(list(self.B))
            self.I.remove_ids(np.array([idx]))
            self.B.remove(idx)
        
        n = len(self.S)
        synced = self.config['cache_sync_off'] or (kwargs['synced'] if 'synced' in kwargs else False)
        if (not synced) and (self.config['cache_sync']>0):
            s_c = self.info['last_sync']
            n_c = len([d for d in self.D[s_c:] if d in ['dev', 'valid']])
            if n_c >= self.config['cache_sync']:
                print("Cache Synchronization Triggered.")
                self.conf = self.get_confidence_distribution()
                self.info['last_sync'] = n
        
        # TODO: currently it saves on every update, where it should save on every sync
        self.save()
    
    def get_confidence_distribution(self):
        U = [k for k,d in zip(self.K,self.D) if d in ['dev', 'valid']]
        processed_eval_dataset = [serialize(self.config['name'], instance) for instance in U[max(0,len(U)-self.config['cache_max_size']):]]
        confidences = []
        for s in processed_eval_dataset:
            e = norm_embed(s, self.C, self.T, self.M)
            ips, _ = self.I.search(e, 1)
            confidences.append((1+ips[0][0])/2)
        return sorted(confidences)

    def get_confidence_threshold(self):
        if 'cache_confidence_ratio' in self.config:
            n = len(self.conf); thres = self.conf[min(int(n*self.config['cache_confidence_ratio']),n-1)] if n > 0 else 1.0
        else:
            thres = self.config['cache_confidence_threshold']
        return thres
    
    def run(self, instance):
        s = serialize(self.config['name'], instance)
        e = norm_embed(s, self.C, self.T, self.M)
        if self.I.ntotal == 0:
            return None, None
        ips, idxs = self.I.search(e, 1)
        ip, idx = ips[0][0], idxs[0][0]
        _, pred, conf = self.K[idx], self.V[idx], (1+ip)/2
        return pred, conf
    
    def rag(self, instance, topK=10):
        s = serialize(self.config['name'], instance)
        e = norm_embed(s, self.C, self.T, self.M)
        _, idxs = self.I.search(e, topK)
        return [(self.K[idx] | {self.config['outputs'][0]['name']: self.V[idx]}) for idx in idxs[0]]