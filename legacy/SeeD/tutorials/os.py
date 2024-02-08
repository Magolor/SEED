from seed import *

if __name__=="__main__":
    project = "os"
    project_path = pjoin(get_config('projects_path'), project)
    CreateProject(project, clear=True)
    args = get_seed_args()
    
    env = Environment()
    env.add(ScriptCell(code="from seed import *"))
    env.add(ScriptCell(code=
f"""
records = LoadJson('./data/OS.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='label', sample={512}, reorder="{args.reorder}", task="topic_classification", batch_size={args.batch_size}, seed={args.random_seed})
"""
    ))
    AddModule(project=project,
        name = "topic_classification",
        desc = "Given a tweet message, determine whether it is hatespeech or benign.",
        inputs = {
            "text": "str. The tweet message.",
        },
        output = "int. Output 0 for hatespeech, 1 for benign.",
        examples = [
            {
                "inputs": {
                    "text": "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...",
                },
                "output": 1,
                "info": None,
            },
            {
                "inputs": {
                    "text": "\" &amp; you might not get ya bitch back &amp; thats that \"",
                },
                "output": 0,
                "info": None,
            },
            {
                "inputs": {
                    "text": "\"@CB_Baby24: @white_thunduh alsarabsss\" hes a beaner smh you can tell hes a mexican",
                },
                "output": 0,
                "info": None,
            },
        ],
        code_tools = {
        },
        query_tools = {
        },
        args = args,
    )
    
    env.add(ScriptCell(code=f"clr_llm_count()"))
    
    env.add(ScriptCell(code=
"""
from modules import *
service = topic_classification_integrated_class(Data(
    {args}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [int(x) for x in service.responses]
Delete("{output_file}",rm=True); CreateFile("{output_file}")
evaluate_binary("{output_file}", responses, labels)
evaluate_llm_count("{output_file}")
""".format(args=repr(args),output_file=f"./{args.identifier}.txt")
    ))
    
    env.export(pjoin(project_path, "main.py"))