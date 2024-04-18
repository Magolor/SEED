# %%
from pyheaven import *
import numpy as np

confs = LoadJson("confs.jsonl", backend="jsonl")
CACHE_CONF = {
    c/10.0: np.percentile(confs, c*10) for c in range(11)
} | {None: 0.0}
model_confs = LoadJson("model_confs.jsonl", backend="jsonl")
MODEL_CONF = {
    c/10.0: np.percentile(model_confs, c*10) for c in range(11)
} | {None: 0.0}

def run_pipeline(
    record,
    order,
    llm_mode,
    cache_conf,
    model_conf,
    code_ensemble,
    default_result = None
):
    c = 0
    p_cache_conf = CACHE_CONF[cache_conf]
    p_model_conf = MODEL_CONF[model_conf]
    for module_name in order:
        if module_name == "llm":
            result = record[f'llm_{llm_mode}']
        elif module_name == "cache":
            result = record[f'cache'] if record['cache_conf'] >= p_cache_conf else None
        elif module_name == "model":
            result = record[f'lm'] if record['lm_conf'] >= p_model_conf else None
        else:
            result = record['code_e'] if code_ensemble else record['code_3']
        if module_name=="llm":
            c += 1
        if result is not None:
            return result, c
    return default_result, c

import evaluate
f1_metric = evaluate.load("f1")
def evaluate_f1(gt, pd):
    f1 = f1_metric.compute(references=gt, predictions=pd)
    return f1['f1']

X = 0
def eval_pipeline(
    records,
    pipeline
):
    order, llm_mode, cache_conf, model_conf, code_ensemble = pipeline
    gts = [record['is_same'] for record in records]
    pds_c = [run_pipeline(record, order, llm_mode, cache_conf, model_conf, code_ensemble, default_result=0) for record in records]
    pds = [pd for pd, c in pds_c]; C = sum([c for pd, c in pds_c])
    global X
    X += 1
    return evaluate_f1(gts, pds), C

# %%
# Brute Force
G = 0.05
modules = ['llm', 'cache', 'model', 'code']
def bf_hyper(module_name, plan):
    assert (module_name in ['llm', 'cache', 'model', 'code'])
    if module_name == "llm":
        for llm_mode in ['rag', 'sample', 'balanced']:
            yield (tuple(list(plan[0])+[module_name]), llm_mode, plan[2], plan[3], plan[4])
    if module_name == "cache":
        for cache_conf in [c/10.0 for c in range(11)]:
            yield (tuple(list(plan[0])+[module_name]), plan[1], cache_conf, plan[3], plan[4])
    if module_name == "model":
        for model_conf in [c/10.0 for c in range(11)]:
            yield (tuple(list(plan[0])+[module_name]), plan[1], plan[2], model_conf, plan[4])
    if module_name == "code":
        for code_ensemble in [True, False]:
            yield (tuple(list(plan[0])+[module_name]), plan[1], plan[2], plan[3], code_ensemble)

# %%
records = LoadJson("data/Abt-Buy_results.jsonl", backend="jsonl")
plans = dict()
empty_plan = (tuple(), None, None, None, None)
plans[empty_plan] = eval_pipeline(records, empty_plan)
T = 0
while True:
    print(f"Iteration #{T}"); T += 1
    new_plans = {plan:(f1,cost) for plan,(f1,cost) in plans.items()}
    for plan in TQDM(plans):
        for module_name in modules:
            if module_name not in plan[0]:
                for new_plan in bf_hyper(module_name, plan):
                    new_plans[new_plan] = eval_pipeline(records, new_plan)
    new_plans = {
        plan:(f1,cost) for plan,(f1,cost) in sorted(new_plans.items(), key=lambda x: x[1][0], reverse=True)
        if all([(
            not (
                (f1_ > f1) and (cost_ < cost)
                # ((f1_ > f1) and (cost_ <= cost)) or (f1_ >= f1 and cost_ < cost) or (f1_ >= f1 and cost_ <= cost and len(p_[0]) < len(plan[0]))
            )
        ) for p_,(f1_,cost_) in new_plans.items()])
    }
    br = True
    for new_plan in new_plans:
        if new_plan not in plans:
            br = False
            break
    if br:
        break
    plans = {plan:(f1,cost) for plan,(f1,cost) in new_plans.items()}
    print(len(plans))
    best_plan = max(plans.items(), key=lambda x: x[1][0])
    best_f1 = best_plan[1][0]
    print(best_plan)
    valid_plans = {plan:(f1,cost) for plan,(f1,cost) in plans.items() if f1>=best_f1-G}
    cheapest_plan = min(valid_plans.items(), key=lambda x: (x[1][1], -x[1][0], len(x[0][0])))
    print(cheapest_plan)
    print(X)

# %%
SaveJson(plans, "bf_plans.jsonl", backend='jsonl')
print(len(plans))
best_plan = max(plans.items(), key=lambda x: x[1][0])
best_f1 = best_plan[1][0]
print(best_plan)
valid_plans = {plan:(f1,cost) for plan,(f1,cost) in plans.items() if f1>=best_f1-G}
cheapest_plan = min(valid_plans.items(), key=lambda x: (x[1][1], -x[1][0], len(x[0][0])))
print(cheapest_plan)
print(X)