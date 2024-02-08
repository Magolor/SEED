# SEED: Domain-Specific Data Curation With Large Language Models

This is the code repo for the paper "SEED: Domain-Specific Data Curation With Large Language Models".

SEED is an approach that leverages Large Language Models (LLMs) to automatically generate domain-specific data curation solutions. By describing a task, input data, and expected output, the SEED compiler produces an executable pipeline consisting of LLM-generated code, small models, and data access modules.

## Info

**Current Version**: v0.1.0

**Compatibility**: Tested on MacBook M1 Pro, MacOS 12.6.2, Python 3.9, Pytorch 1.13.1

**Hardware Requirements**:  None

## Installation

**Notice that the code is currently undergoing reconstruction and unable to install!**

First install the prerequisites and the SEED package.

```bash
git clone git@github.com:Magolor/SEED.git
cd ./SEED/legacy/SeeD/
pip install -r requirements.txt
pip install -e .
```

Then run the installer to initialize configurations (including OpenAI keys):
```python
python installer.py
```

## Quickstart

The legacy version requires the user to manually write short programs to connect the modules. Please checkout `demo_projects` and `tutorials` for examples.

Use the restaurant as an example, to build a project from scratch:

1. Create a `restaurant` project folder and prepare data (`restaurant/data/Restaurant.jsonl`) as a `.jsonl` file, where each line of the file corresponds to a data record.
2. Create a script to load data, for example:
```python
args = get_seed_args()
records = LoadJson('./data/Restaurant.jsonl',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='city', sample=512, reorder="RND", task="data_imputation", batch_size=1, seed=42)
```
3. Use `AddModule` to configure the modules and arguments:
```python
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
```
The generated code will be placed under the `module` folder within the project folder.
4. After the module is added, the service API could be created based on the `name` argument in `AddModule` and should be used as follows:
```python
service = data_imputation_integrated_class(Data(
    {args}
))
```
where `{args}` should be the arguments dictionary obtained from `get_seed_args()`.

The `restaurant` project is mainly for code generation, for models please refer to `os`, for tools please refer to `ott`. Notice that the as the `projects_path` is configured as `"./projects/"` by default, run `python tutorials/*.py` instead of `cd ./tutorials/ && python *.py`.