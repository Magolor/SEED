from ..templates import *
from .agent import Agent

class ModelAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = pjoin(self.config['project_path'], "agents", "model")
        self.model = load_model(self.config['model_initial_ckpt'], model_type=eval(self.config['model_type']), **self.config['model_args'])
        self.C = self.model[0]; self.T = self.model[1]; self.M = self.model[2]
        self.load()
    
    def load_model(self):
        self.model = load_model(pjoin(self.path, "ckpt"), model_type=eval(self.config['model_type']), **self.config['model_args'])
        self.C = self.model[0]; self.T = self.model[1]; self.M = self.model[2]
    
    def load(self, init=False):
        self.info = LoadJson(pjoin(self.path, "info.json"))                             if (not init) and ExistFile(pjoin(self.path, "info.json"))         else dict({'last_sync_small': 0, 'last_sync_large': 0, 'last_sync_confi': 0})
        self.S = set(LoadJson(pjoin(self.path, "identifiers.jsonl"), backend="jsonl"))  if (not init) and ExistFile(pjoin(self.path, "identifiers.jsonl")) else set()
        self.K = LoadJson(pjoin(self.path, "keys.jsonl"), backend="jsonl")              if (not init) and ExistFile(pjoin(self.path, "keys.jsonl"))        else list()
        self.V = LoadJson(pjoin(self.path, "values.jsonl"), backend="jsonl")            if (not init) and ExistFile(pjoin(self.path, "values.jsonl"))      else list()
        self.D = LoadJson(pjoin(self.path, "data_split.jsonl"), backend="jsonl")        if (not init) and ExistFile(pjoin(self.path, "data_split.jsonl"))  else list()
        self.conf = LoadJson(pjoin(self.path, "confidences.jsonl"), backend="jsonl")    if (not init) and ExistFile(pjoin(self.path, "confidences.jsonl")) else self.get_confidence_distribution()
    
    def save(self):
        SaveJson(self.info, pjoin(self.path, "info.json"))
        SaveJson(sorted(list(self.S)), pjoin(self.path, "identifiers.jsonl"), backend="jsonl")
        SaveJson(self.K, pjoin(self.path, "keys.jsonl"), backend="jsonl")
        SaveJson(self.V, pjoin(self.path, "values.jsonl"), backend="jsonl")
        SaveJson(self.D, pjoin(self.path, "data_split.jsonl"), backend="jsonl")
        SaveJson(self.conf, pjoin(self.path, "confidences.jsonl"), backend="jsonl")
    
    def compile(self):
        CreateFolder(self.path); CreateFile(pjoin(self.path,'__init__.py'))
        with open(pjoin(self.path,'__init__.py'), 'w') as f:
            f.write(format_model_code(**self.config))
        self.load(init=True); self.save()
        save_model(pjoin(self.path, "ckpt"), self.C, self.T, self.M)
    
    def update(self, instance, label, **kwargs):
        if ignore_label(label): label="nan"
        s = serialize(self.config['name'], instance)
        if s in self.S:
            return
        
        self.S.add(s)
        self.K.append(instance)
        self.V.append(label)
        self.D.append("dev" if ignore_label(label) else ("train" if random.random() < 0.8 else "valid"))
        
        n = len(self.S)
        synced = self.config['model_sync_off'] or (kwargs['synced'] if 'synced' in kwargs else False)
        if (not synced) and (self.config['model_sync_large']>0):
            s_l = self.info['last_sync_large']
            n_l = len([d for d in self.D[s_l:] if d == "train"])
            if n_l >= self.config['model_sync_large']:
                print("Model Large Synchronization Triggered.")
                T = [(k,v) for k,v,d in zip(self.K,self.V,self.D) if d=="train"]
                E = [(k,v) for k,v,d in zip(self.K,self.V,self.D) if d=="valid"]
                self.model_sync(T, E)
                self.info['last_sync_large'] = n
                self.info['last_sync_small'] = n
                synced = True
            else:
                synced = False
        if (not synced) and (self.config['model_sync_small']>0):
            s_s = self.info['last_sync_small']
            n_s = len([d for d in D[s_s:] if d == "train"])
            if n_s >= self.config['model_sync_small']:
                print("Model Small Synchronization Triggered.")
                T = [(k,v) for k,v,d in zip(self.K[s_s:],self.V[s_s:],self.D[s_s:]) if d=="train"]
                E = [(k,v) for k,v,d in zip(self.K,self.V,self.D) if d=="valid"]
                self.model_sync(T, E, training_args={
                    'num_train_epochs': 1,
                    'lr_scheduler_type': None,
                })
                self.info['last_sync_small'] = n
                synced = True
            else:
                synced = False
        
        if (self.config['model_sync_confi']>0):
            s_c = self.info['last_sync_confi']
            n_c = len([d for d in self.D[s_c:] if d in ['dev', 'valid']])
            if n_c >= self.config['model_sync_confi']:
                print("Model Confidence Synchronization Triggered.")
                self.conf = self.get_confidence_distribution()
                self.info['last_sync_confi'] = n
                synced = True
                
        # TODO: currently it saves on every update, where it should save on every sync
        self.save()
        
    def model_sync(self, T, E, training_args=dict()):
        processed_train_dataset = [[serialize(self.config['name'], instance), label] for instance, label in T]
        processed_eval_dataset = [[serialize(self.config['name'], instance), label] for instance, label in E]
        finetune(
            train_dataset = processed_train_dataset,
            eval_dataset = processed_eval_dataset,
            ckpt = pjoin(self.path, "ckpt"), 
            model_type_str = self.config['model_type'],
            evaluation_metric = self.config['evaluation_metric'],
            training_args = training_args,
        )
        self.load_model()
    
    def get_confidence_distribution(self):
        U = [k for k,d in zip(self.K,self.D) if d in ['dev', 'valid']]
        processed_eval_dataset = [serialize(self.config['name'], instance) for instance in U[max(0,len(U)-self.config['model_max_size']):]]
        confidences = []
        for s in processed_eval_dataset:
            _, confidence = model_operation(s, self.C, self.T, self.M, model_type_str=self.config['model_type'])
            confidences.append(confidence)
        return sorted(confidences)

    def get_confidence_threshold(self):
        if 'model_confidence_ratio' in self.config:
            n = len(self.conf); thres = self.conf[min(int(n*self.config['model_confidence_ratio']),n-1)] if n > 0 else 1.0
        else:
            thres = self.config['model_confidence_threshold']
        return thres
    
    def run(self, instance):
        s = serialize(self.config['name'], instance)
        pred, conf = model_operation(s, self.C, self.T, self.M, model_type_str=self.config['model_type'])
        if 'verbalizer' in self.config['outputs'][0]:
            pred = self.config['outputs'][0]['verbalizer'][pred]
        return pred, conf