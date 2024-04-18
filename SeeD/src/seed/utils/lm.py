from .utils import *
from .metric import get_evaluation_metric

def tokenize(s, T):
    return T(s, padding=True, truncation=True, return_tensors='pt')
def embed(s, C, T, M):
    with torch.no_grad():
        return M(**tokenize(s, T)).last_hidden_state.mean(dim=1).squeeze()
def generate(s, C, T, M, max_length=128):
    with torch.no_grad():
        return T.decode(M.generate(**tokenize(s, T), max_length=max_length)[0], skip_special_tokens=True)
def classify(s, C, T, M):
    with torch.no_grad():
        return M(**tokenize(s, T)).logits.softmax(dim=1).tolist()[0]
def perplexity(x, y, C, T, M):
    with torch.no_grad():
        return M(**tokenize(x, T), labels=tokenize(y, T)['input_ids']).loss.exp().item()
def norm_embed(s, C, T, M):
    e = embed(s, C, T, M).cpu().detach().numpy().reshape(1, -1); faiss.normalize_L2(e); return e

def model_operation(s, C, T, M, model_type_str):
    if model_type_str == 'AutoModelForSequenceClassification':
        logits = classify(s, C, T, M)
        pred = logits.index(max(logits))
        confidence = np.clip(logits[pred]-1/len(logits), 0., 1.)
        return pred, confidence
    if model_type_str == 'AutoModelForSeq2SeqLM':
        pred = generate(s, C, T, M)
        confidence = np.clip(1/perplexity(s, pred, C, T, M), 0., 1.)
        return pred, confidence
    raise NotImplementedError

def encode_dataset(data, T, model_type_str):
    if model_type_str == 'AutoModelForSequenceClassification':
        df = pd.DataFrame(data, columns=[  'text',  'label']).astype({'text': str, 'label': int})
        def encode_instance(instance):
            return tokenize(instance['text'], T)
    if model_type_str == 'AutoModelForSeq2SeqLM':
        df = pd.DataFrame(data, columns=['source', 'target']).astype({'source': str, 'target': str})
        def encode_instance(instance):
            return tokenize(instance['source'], T)
    if model_type_str == 'AutoModelForTextEncoding':
        df = pd.DataFrame(data, columns=[ 'text']).astype({'text': str})
        def encode_instance(instance):
            return tokenize(instance['text'], T)
        raise NotImplementedError
    return Dataset.from_pandas(df).map(encode_instance, batched=True)

def finetune(train_dataset, eval_dataset, ckpt, model_type_str,
             evaluation_metric = 'accuracy',
             model_tune = False,
             model_eval = False,
             debug = False,
             training_args = dict(),
             **kwargs):
    C, T, _ = load_model(ckpt, model_type=eval(model_type_str), **kwargs)
    encoded_train_dataset = encode_dataset(train_dataset, T, model_type_str)
    encoded_eval_dataset = encode_dataset(eval_dataset, T, model_type_str)
    if debug:
        encoded_train_dataset = encoded_train_dataset.select(range(128))
        encoded_eval_dataset = encoded_eval_dataset.select(range(128))
    
    metric = get_evaluation_metric(evaluation_metric)
    
    def compute_metrics(eval_pred):
        # For T5ForSequenceClassification
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        logits = predictions[0]; preds = np.argmax(logits, axis=-1)
        # # For BertForSequenceClassification
        # predictions, labels = eval_pred
        # logits = predictions; preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = merge_dicts([{
        "output_dir": TEMP_TRAINING_PATH,
        "num_train_epochs": 30,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "optim": 'adamw_torch',
        "lr_scheduler_type": 'cosine',
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 1,
        "eval_accumulation_steps": 8,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
    }, training_args])
    training_args = {k:v for k,v in training_args.items() if v is not None}
    training_args = TrainingArguments(**training_args)
    def model_init():
        _, _, M = load_model(ckpt, model_type=eval(model_type_str), **kwargs); return M
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    trainer = Trainer(
        tokenizer = T,
        model = None,
        model_init = model_init,
        args = training_args,
        train_dataset = encoded_train_dataset,
        eval_dataset = encoded_eval_dataset,
        compute_metrics = compute_metrics,
        callbacks = [early_stopping],
    )
    
    if model_tune:
        # TODO: check hyperparameter search support
        def ray_hp_space(trial):
            return {
                "learning_rate": tune.loguniform(1e-7, 1e-4),
                "weight_decay": tune.choice([0, 0.01, 0.02]),
            }
        trainer.hyperparameter_search(
            direction = "maximize", 
            backend = "ray", 
            n_trials = 10,
            hp_space = ray_hp_space,
        )
    else:
        trainer.train()
    
    if model_eval:
        profile = trainer.evaluate()
    else:
        profile = None
    save_model(ckpt, C, T, trainer.model, profile=profile)
