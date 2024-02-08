from seed import *

def get_<<api.name>>_api(agent):
    if agent == 'cache':
        from agents.cache import <<api.name>>
        return <<api.name>>
    elif agent == 'codeg':
        from agents.codeg import <<api.name>>
        return <<api.name>>
    elif agent == 'model':
        from agents.model import <<api.name>>
        return <<api.name>>
    elif agent == 'llmqa':
        from agents.llmqa import <<api.name>>
        return <<api.name>>
    else:
        raise NotImplementedError

<<api.api_def(with_kwargs=True)>>
    config = LoadJson("config.json")
    instance = dict(<<api.asgs>>)
    final_response = None; final_source = None
    if 'groundtruth' in kwargs:
        final_response = kwargs['groundtruth']; final_source = 'groundtruth'
    p_fallback = config['p_fallback']
    force_fallback = config['activate_llmqa'] and (np.random.random()<p_fallback)
    if force_fallback:
        print(f"forced fallback to llmqa (p={p_fallback*100:.1f}%)")
    modules = ['cache', 'model', 'codeg', 'llmqa']
    for module in modules:
        if (final_source is None) and config[f'activate_{module}'] and (not force_fallback):
            try:
                response = get_<<api.name>>_api(module)(<<api.asgs>>)
                print(f"{module} response:", response)
                if response is not None:
                    final_response = response; final_source = module
            except Exception as e:
                print(e)
    if (final_source is None):
        final_response = config['outputs'][0]['default']; final_source = 'default'
    if final_source in ['groundtruth', 'llmqa']:
        if config['activate_cache']:
            from agents.cache import get_cache_agent
            get_cache_agent().update(instance=instance, label=final_response, synced=False)
        if config['activate_model']:
            from agents.model import get_model_agent
            get_model_agent().update(instance=instance, label=final_response, synced=False)
    if final_source in ['default', 'cache', 'codeg', 'model']:
        if config['activate_cache']:
            from agents.cache import get_cache_agent
            get_cache_agent().update(instance=instance, label="nan", synced=False)
        if config['activate_model']:
            from agents.model import get_model_agent
            get_model_agent().update(instance=instance, label="nan", synced=False)
    if 'profiler' in kwargs:
        kwargs['profiler'][final_source] += 1
    return final_response