# %%
from seed import *

# %%

records = LoadJson('./data/Restaurant.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='city', sample=512, reorder="RND", task="data_imputation", batch_size=32, seed=42)


# %%
clr_llm_count()

# %%

from modules import *
service = data_imputation_integrated_class(Data(
    {'use_cache': False, 'cache_threshold': 0.3, 'use_codegen': True, 'use_ensemble': True, 'timeout': 5, 'use_logical_eval': False, 'use_example_eval': False, 'use_testgen_eval': False, 'use_simul': False, 'custom_checkpoint': None, 'simul_type': 'clslm', 'simul_num_classes': 2, 'simul_threshold': 0.6, 'use_tools': False, 'use_batch': False, 'batch_size': 32, 'no_query': False, 'reorder': 'RND', 'random_seed': 42, 'identifier': 'base'}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [str(x) for x in service.responses]
Delete("./base.txt",rm=True); CreateFile("./base.txt")
evaluate_fuzzy("./base.txt", responses, labels, thres=75)
evaluate_llm_count("./base.txt")


# %%
