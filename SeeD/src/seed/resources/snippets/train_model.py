# %%
from seed import *

# %%
name = "<<project_name>>"
train_data_path = "<<api.config['labelled_path']>>"
train_dataset = [({k:v for k,v in r.items() if k!='id' and k!='is_same'}, r['is_same']) for r in LoadJson(train_data_path, backend="jsonl")]
processed_train_dataset = [[serialize(name, instance), label] for instance, label in train_dataset]; del train_dataset

eval_data_path = "<<api.config['examples_path']>>"
eval_dataset = [({k:v for k,v in r.items() if k!='id' and k!='is_same'}, r['is_same']) for r in LoadJson(eval_data_path, backend="jsonl")]
processed_eval_dataset = [[serialize(name, instance), label] for instance, label in eval_dataset]; del eval_dataset

# %%
CreateFolder("ckpts/<<project_name>>")
C, T, M = load_model(
    ckpt = "t5-small",
    model_type = AutoModelForSequenceClassification
)
save_model("./ckpts/<<project_name>>", C, T, M)

# %%
finetune(
    train_dataset = processed_train_dataset,
    eval_dataset = processed_eval_dataset,
    ckpt = "./ckpts/<<project_name>>",
    model_type_str = "AutoModelForSequenceClassification",
    evaluation_metric = 'f1',
    
    model_tune = False,
    model_eval = True,
    debug = False
)

# %%