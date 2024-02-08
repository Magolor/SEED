from seed import *
from __init__ import <<api.name>>, get_<<api.name>>_api

import warnings
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def evaluate_<<api.name>>():
    train_data_path = "<<api.config['labelled_path']>>"
    train_data = LoadJson(train_data_path, backend="jsonl") if ExistFile(train_data_path) else []
    if <<debug>>>0:
        train_data = train_data[:<<debug>>]
    if (train_data_path is not None) and ExistFile(train_data_path):
        for i, record in enumerate(TQDM(train_data)):
            result = <<api.name>>(
                **record,
                groundtruth = record["<<api.output>>"]
            )

    evaluation_path = "<<api.config['evaluation_path']>>"
    if ExistFile(evaluation_path):
        eval_data = LoadJson(evaluation_path, backend="jsonl")
        if <<debug>>>0:
            eval_data = eval_data[:<<debug>>]
        metric = get_evaluation_metric("<<api.config['evaluation_metric']>>")
        gts, pds = list(), list()
        profiler = defaultdict(int)
        pbar = TQDM(eval_data)
        for record in pbar:
            result = <<api.name>>(
                **record,
                profiler = profiler
            )
            gts.append(record["<<api.output>>"])
            pds.append(result)
            m = metric.compute(references=gts, predictions=pds)["<<api.config['evaluation_metric']>>"]
            pbar.set_description(f"<<api.config['evaluation_metric']>>: {m:.4f}")
            profiler["<<api.config['evaluation_metric']>>"] = m
            SaveJson(profiler, "profile.json", indent=4)