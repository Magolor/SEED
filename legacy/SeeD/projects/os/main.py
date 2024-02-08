# %%
from seed import *

# %%

records = LoadJson('./data/OS.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='label', sample=512, reorder="FAR", task="topic_classification", batch_size=32, seed=42)


# %%
clr_llm_count()

# %%

from modules import *
service = topic_classification_integrated_class(Data(
    {'use_cache': False, 'cache_threshold': 0.3, 'use_codegen': False, 'use_ensemble': False, 'timeout': 5, 'use_logical_eval': False, 'use_example_eval': False, 'use_testgen_eval': False, 'use_simul': False, 'simul_type': 'seqlm', 'simul_num_classes': 2, 'simul_threshold': 0.6, 'use_batch': True, 'batch_size': 32, 'no_query': False, 'reorder': 'FAR', 'random_seed': 42, 'identifier': 'exp_o=FAR_b=32'}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [int(x) for x in service.responses]
Delete("./exp_o=FAR_b=32.txt",rm=True); CreateFile("./exp_o=FAR_b=32.txt")
evaluate_binary("./exp_o=FAR_b=32.txt", responses, labels)
evaluate_llm_count("./exp_o=FAR_b=32.txt")


# %%
