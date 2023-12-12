from .utils import *

class LMSimulator(object):
    def __init__(self, path, init=False, sync_updates=64):
        self.path = path
        if init: self.init(model=init)
        else: self.load()
        self.sync_updates = sync_updates; self.sync = 0

    def load(self):
        try:
            self.config = LoadJson(pjoin(self.path, "config.json"))
            self.tokenizer, self.model = load_model(self.config['model_path'])
            self.dataset = LoadJson(self.config["dataset_path"], backend="jsonl")
        except:
            self.init(model="bert-large-uncased")
    
    def save(self):
        SaveJson(self.config, pjoin(self.path, "config.json"), indent=4)
        self.tokenizer.save_pretrained(self.config["model_path"])
        self.model.save_pretrained(self.config["model_path"])
        SaveJson(self.dataset, self.config["dataset_path"], backend="jsonl")
    
    def init(self, model):
        self.config = {"model": model, "dim": 384}
        self.config["model_path"] = pjoin(self.path,"ckpt")
        self.config["dataset_path"] = pjoin(self.path, "dataset.jsonl")
        CreateFolder(self.config["model_path"])
        CreateFile(self.config["dataset_path"])
        self.tokenizer, self.model = load_model(self.config['model'], download=True)
        self.dataset = list()
        self.save()
    
    def update(self, key, value):
        self.dataset.append({'source':key, 'target':value}); self.sync += 1
        if self.sync >= self.sync_updates:
            self.sync = 0; self.train(); self.save()

    def encode(self, example):
        return self.tokenizer(example['source'], truncation=True, padding='max_length')
    
    def query(self, key):
        return inference(key, self.tokenizer, self.model)
    
    def train(self, cont=False):
        if not cont:
            self.tokenizer, self.model = load_model(self.config['model'])
        
        encoded_dataset = create_dataset(self.dataset).map(self.encode, batched=True)
        dataset_split = encoded_dataset.train_test_split(test_size=0.1)
        train_dataset, valid_dataset = dataset_split['train'], dataset_split['test']
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.path,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=2,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(
            tokenizer=self.tokenizer,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
            
class CLSSimulator(LMSimulator):
    def __init__(self, path, init=False, sync_updates=32):
        self.path = path
        if init: self.init(model=init)
        else: self.load()
        self.sync_updates = sync_updates; self.sync = 0

    def load(self):
        try:
            self.config = LoadJson(pjoin(self.path, "config.json"))
            self.tokenizer, self.model = load_model(self.config['model_path'], model_type=AutoModelForSequenceClassification)
            self.dataset = LoadJson(self.config["dataset_path"], backend="jsonl")
        except:
            self.init(model="bert-large-uncased")
    
    def init(self, model):
        self.config = {"model": model, "num_classes": 2}
        self.config["model_path"] = pjoin(self.path,"ckpt")
        self.config["dataset_path"] = pjoin(self.path, "dataset.jsonl")
        CreateFolder(self.config["model_path"])
        CreateFile(self.config["dataset_path"])
        self.tokenizer, self.model = load_model(self.config['model'], download=True, model_type=AutoModelForSequenceClassification, num_labels=self.config["num_classes"])
        self.dataset = list()
        self.save()
    
    def update(self, key, value):
        self.dataset.append({'text':key, 'label':value}); self.sync += 1
        if self.sync >= self.sync_updates:
            self.sync = 0; self.train(); self.save()
        
    def encode(self, example):
        return self.tokenizer(example['text'], truncation=True, padding='max_length')
    
    def query(self, key):
        return classify(key, self.tokenizer, self.model)
    
    def train(self, cont=False):
        print('training triggered')
        if not cont:
            self.tokenizer, self.model = load_model(self.config['model'], model_type=AutoModelForSequenceClassification, num_labels=self.config["num_classes"])
        
        encoded_dataset = create_dataset(self.dataset).map(self.encode, batched=True)
        dataset_split = encoded_dataset.train_test_split(test_size=0.1)
        train_dataset, valid_dataset = dataset_split['train'], dataset_split['test']
        data_collator = DefaultDataCollator()
        
        training_args = TrainingArguments(
            output_dir=self.path,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=3,
        )

        trainer = Trainer(
            tokenizer=self.tokenizer,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()