from .utils import *
from .agents import *
from .templates import *

def CreateProject(name, workspace="./projects/"):
    '''
    Create a new project with the given name and path. Notice that the config file is initialized with the default config, which should be modified and completed by the user.
    Args:
        name: str. Name of the project.
        workspace: str. Path of the workspace, that is, the created project will be placed in `<workspace>/<name>`.
    '''
    CreateFolder(workspace); root = pjoin(workspace, name)
    CreateFolder(root); CreateFolder(pjoin(root, 'agents')); CreateFolder(pjoin(root, 'data'))
    config = LoadJson(find_resource('configs/default_project.json')); config['project_name'] = name
    SaveJson(config, pjoin(workspace, name, 'config.json'), indent=4)
    SaveJson(LoadJson(find_resource('configs/hyperparameters.json')), pjoin(workspace, name, 'hyperparameters.json'), indent=4)

def CompleteConfig(path):
    '''
    Automatically complete and examine the config file of the project.
    This is only for convenience and does NOT serve as a substitute for manually completing the config file.
    Notice that it modifies the layout of the config file.
    Args:
        path: str. Path of the project.
    '''
    config = LoadJson(pjoin(path, 'config.json'))
    config = merge_dicts([LoadJson(find_resource('configs/complete_project.json')), config])
    config['project_path'] = os.path.abspath(path)
    assert (config['name']), "`name` must be specified in `config.json`!"
    assert (config['task_desc']), "`task_desc` must be specified in `config.json`!"
    assert (config['evaluation_path']), "`evaluation_path` must be specified in `config.json`!"
    assert (config['evaluation_metric']), "`evaluation_metric` must be specified in `config.json`!"
    assert (len(config['outputs']) == 1), "Currently, only one output is supported! You may create multiple projects for multiple outputs."
    if (not os.path.isabs(config['evaluation_path'])):
        config['evaluation_path'] = os.path.abspath(pjoin(path, config['evaluation_path']))
    if (not config['examples_path']):
        config['examples_path'] = config['evaluation_path']
    elif (not os.path.isabs(config['examples_path'])):
        config['examples_path'] = os.path.abspath(pjoin(path, config['examples_path']))
    if ('verbalizer' in config['outputs'][0]):
        config['outputs'][0]['deverbalizer'] = {v:i for i,v in enumerate(config['outputs'][0]['verbalizer'])}
        if config['activate_model']:
            config['model_args']['num_labels'] = len(config['outputs'][0]['verbalizer'])
    SaveJson(config, pjoin(path, 'config.json'), indent=4)

def CompileProject(path):
    '''
    Compile a project and generating a runnable API from the config file.
    Args:
        path: str. Path of the project.
    '''
    CompleteConfig(path)
    config = LoadJson(pjoin(path, 'config.json'))
    
    if True: # For debugging only
        agent_empty = Agent(**config); agent_empty.compile()
    
    if config['activate_cache']:
        agent_cache = CacheAgent(**config); agent_cache.compile()
    if config['activate_model']:
        agent_model = ModelAgent(**config); agent_model.compile()
    if config['activate_codeg']:
        codeg_examples = GetExamples(path, agent="codeg")
        codev_examples = GetExamples(path, agent="codev")
        agent_model = CodeGenAgent(**config); agent_model.compile(codeg_examples=codeg_examples, codev_examples=codev_examples)
    if config['activate_tools']:
        raise NotImplementedError
    if config['activate_llmqa']:
        agent_llmqa = LLMQAAgent(**config); agent_llmqa.compile()
        
    with open(pjoin(path, "__init__.py"), "w") as f:
        f.write(format_pipeline_code(**config))

    with open(pjoin(path, "evaluation.py"), "w") as f:
        f.write(format_evaluation_code(**config))

    with open(pjoin(path, "train_model.py"), "w") as f:
        f.write(format_train_model_code(**config))

    with open(pjoin(path, "agents", "__init__.py"), "w") as f:
        pass

def GetExamples(path, agent="", instance=None):
    '''
    Retrieve examples of the project. If RAG examples are to be retrieved, the cache agent must be activated and the 'instance' must be specified.
    Args:
        path: str. Path of the project.
        agent: str. Which agent to retrieve examples for. For example "codeg" for code generation agent (currently only "codeg" and "codev" use this). Default as "" for general examples.
        instance: dict. A specific query instance. Only required for RAG examples. Default as `None`.
    Return:
        List[dict]. A list of examples. Each item is a dict with the following keys:
            id: str. ID of the example. Default as `None`.
            inputs: dict. Inputs of the example.
            outputs: dict. Outputs of the example.
    '''
    config = LoadJson(pjoin(path, 'config.json'))
    prefix = f'{agent}_examples' if agent else 'examples'
    if (f'{prefix}_manual' in config) and (config[f'{prefix}_manual']):
        data = config[f'{prefix}_manual']
    elif (f'{prefix}_path' in config) and (config[f'{prefix}_path']):
        data = LoadJson(config[f'{prefix}_path'], backend='jsonl')
    else:
        data = []
    
    np.random.seed(42)
    if config[f'{prefix}_mode'] == 'all':
        examples = data
    elif config[f'{prefix}_mode'] == 'sample':
        examples = np.random.choice(data, size=min(len(data), config[f'{prefix}_count']), replace=False)
    elif config[f'{prefix}_mode'] == 'balanced':
        values = set([e[config['outputs'][0]['name']] for e in data])
        n = config[f'{prefix}_count']//len(values); examples = list()
        for v in values:
            v_data = [e for e in data if e[config['outputs'][0]['name']] == v]
            examples.extend(np.random.choice(v_data, size=min(len(v_data), n), replace=False))
    elif config[f'{prefix}_mode'] == 'rag':
        if instance is None:
            examples = list()
        else:
            # assert (instance is not None), "Instance is required for RAG examples! (pass in 'instance' argument)"
            assert (config['activate_cache']), "Cache agent is required for RAG examples! (set 'activate_cache' to True in config.json)"
            examples = CacheAgent(**config).rag(instance, topK=config[f'{prefix}_count'])
    else:
        raise NotImplementedError
    
    inputs_keys = [i['name'] for i in config['inputs']]
    outputs_keys = [i['name'] for i in config['outputs']]
    return [{
        'id': e['id'] if 'id' in e else None,
        'inputs': {k:v for k,v in e.items() if k in inputs_keys},
        'outputs': {k:v for k,v in e.items() if k in outputs_keys},
    } for e in examples]

def add_config(path, config, profile):
    c = 0
    while ExistFile(pjoin(path, f'configs/{c}.json')):
        c += 1
    SaveJson(config, pjoin(path, f'configs/{c}_config.json'), indent=4)
    SaveJson(profile, pjoin(path, f'configs/{c}_profile.json'), indent=4)

def evaluate_config(path, config):
    print("Evaluating config:",
        f"cache = {config['cache_confidence_ratio'] if config['activate_cache'] else -1},",
        f"model = {config['model_confidence_ratio'] if config['activate_model'] else -1},",
        f"codeg = {1 if config['activate_codeg'] else -1},",
        f"llmqa = {config['examples_mode']}",
    )
    SaveJson(config, pjoin(path, 'config.json'), indent=4)
    CompileProject(path)
    CMD("python -c 'from evaluation import *\nevaluate_entity_resolution()'", wait=True)
    return LoadJson(pjoin(path, 'profile.json'))

def list_config_profiles(path):
    c = 0
    while ExistFile(pjoin(path, f'configs/{c}.json')):
        yield LoadJson(pjoin(path, f'configs/{c}_config.json')), LoadJson(pjoin(path, f'configs/{c}_profile.json'))
        c += 1

def get_best_config(path, evaluation_metric):
    config_profiles = list_config_profiles(path)
    config, profile = max(config_profiles, key=lambda x:x[evaluation_metric])
    return config, profile

def HyperparameterTuning(path):
    '''
    Enumerate a project parameter setting and generate their corresponding profiles. The user can then manually choose the config with the best performance.
    Args:
        path: str. Path of the project.
    '''
    space = LoadJson(pjoin(path, 'hyperparameters.json'))
    base_config = LoadJson(pjoin(path, 'config.json'))
    evaluation_metric = base_config['evaluation_metric']
    CreateFolder(pjoin(path, 'configs'))
    for agent in ['cache', 'model', 'codeg', 'tools', 'llmqa']:
        base_config[f'activate_{agent}'] = False

    # 1. Get the best performance, assuming it is the `llm` one
    for examples_mode in space['examples_mode']:
        vari_config = base_config | {
            'activate_llmqa': True,
            'examples_mode': examples_mode
        }
        profile = evaluate_config(path, vari_config)
        add_config(path, vari_config, profile)
    best_llm_config, best_llm_profile = get_best_config(path, evaluation_metric)
    best_performance = best_llm_profile[evaluation_metric]
    
    # 2. Search for the code component
    if True in space['activate_codeg']:
        vari_config = best_llm_config | {
            'activate_codeg': True,
            'examples_mode': examples_mode
        }
        profile = evaluate_config(path, vari_config)
        add_config(path, vari_config, profile)
        if profile[evaluation_metric] < best_performance - space['performance_gap']:
            performance_threshold = best_performance - space['performance_gap']
            best_configs = [best_llm_config]
        else:
            performance_threshold = profile[evaluation_metric]
            best_configs = [best_llm_config, vari_config]
    else:
        performance_threshold = best_performance - space['performance_gap']
        best_configs = [best_llm_config]
    
    # 2. Search for optimized model hyperparameters greedily
    if True in space['activate_model']:
        for best_config in best_configs:
            for c in sorted(space['model_confidence_ratio'], reverse=True):
                vari_config = best_config | {
                    'activate_model': True,
                    'model_confidence_ratio': c
                }
                profile = evaluate_config(path, vari_config)
                add_config(path, vari_config, profile)
                if profile[evaluation_metric] < performance_threshold:
                    break
                else:
                    best_configs.append(vari_config)
    
    # 3. Search for optimized cache hyperparameters greedily
    if True in space['activate_cache']:
        for config in best_configs:
            for c in sorted(space['cache_confidence_ratio'], reverse=True):
                vari_config = config | {
                    'activate_cache': True,
                    'cache_confidence_ratio': c
                }
                profile = evaluate_config(path, vari_config)
                add_config(path, vari_config, profile)
                if profile[evaluation_metric] < performance_threshold:
                    break