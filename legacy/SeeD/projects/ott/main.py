# %%
from seed import *

# %%

records = LoadJson('./data/OTT-QA.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='tables', sample=200, reorder="RND", task="data_discovery", batch_size=32, seed=42)


# %%
clr_llm_count()

# %%

from modules import *
service = data_discovery_integrated_class(Data(
    {'use_cache': False, 'cache_threshold': 0.3, 'use_codegen': False, 'use_ensemble': False, 'timeout': 5, 'use_logical_eval': False, 'use_example_eval': False, 'use_testgen_eval': False, 'use_simul': False, 'custom_checkpoint': None, 'simul_type': 'clslm', 'simul_num_classes': 2, 'simul_threshold': 0.6, 'use_tools': True, 'use_batch': False, 'batch_size': 32, 'no_query': False, 'reorder': 'RND', 'random_seed': 42, 'identifier': 'base'}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [(list(x) if x else []) for x in service.responses]
SaveJson([d|{'response':r} for d,r in zip(records,responses)], "./base.jsonl", backend='jsonl')
Delete("./base.txt",rm=True); CreateFile("./base.txt")
evaluate_list("./base.txt", responses, labels)
evaluate_llm_count("./base.txt")


# %%
