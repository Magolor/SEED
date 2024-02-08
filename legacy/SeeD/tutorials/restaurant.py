from seed import *

if __name__=="__main__":
    project = "restaurant"
    project_path = pjoin(get_config('projects_path'), project)
    CreateProject(project, clear=True)
    args = get_seed_args()
    
    env = Environment()
    env.add(ScriptCell(code="from seed import *"))
    env.add(ScriptCell(code=
f"""
records = LoadJson('./data/Restaurant.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='city', sample={512}, reorder="{args.reorder}", task="data_imputation", batch_size={args.batch_size}, seed={args.random_seed})
"""
    ))
    AddModule(project=project,
        name = "data_imputation",
        desc = "Given a restaurant's information, deduce the city it is located in.",
        inputs = {
            "name": "str. The name of the restaurant.",
            "addr": "str. The address of the restaurant.",
            "phone": "str. The phone number of the restaurant.",
            "type": "str. The food type of the restaurant.",
        },
        output = "str. The city the restaurant is in.",
        examples = [
            {
                "inputs": {
                    "name": "le chardonnay (los angeles)",
                    "addr": "8284 melrose ave.",
                    "phone": "213-655-8880",
                    "type": "french bistro",
                },
                "output": "los angeles",
                "info": "In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.",
            },
            {
                "inputs": {
                    "name": "matsuhisa",
                    "addr": "129 n. la cienega blvd.",
                    "phone": "310/659-9639",
                    "type": "asian",
                },
                "output": "beverly hills",
                "info": "Phone number has area code 310 which represents Beverly Hills, California.",
            },
        ],
        code_tools = {
            "python packages": "You can use any python packages you want. You do not need to install but only import them before using. You can not use supervised-learning method as there is no training data. Though, you can use frozen models if you want.",
        },
        query_tools = {
        },
        args = args,
    )
    
    env.add(ScriptCell(code=f"clr_llm_count()"))
    
    env.add(ScriptCell(code=
"""
from modules import *
service = data_imputation_integrated_class(Data(
    {args}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [str(x) for x in service.responses]
Delete("{output_file}",rm=True); CreateFile("{output_file}")
evaluate_fuzzy("{output_file}", responses, labels, thres=75)
evaluate_llm_count("{output_file}")
""".format(args=repr(args),output_file=f"./{args.identifier}.txt")
    ))
    
    env.export(pjoin(project_path, "main.py"))