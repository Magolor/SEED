from ..utils import *

def format_default_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/default_agent.py")),
        api = api,
        debug = debug,
    )    

def format_llmqa_single_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/llmqa_single.py")),
        api = api,
        debug = debug,
    )

def format_cache_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/cache.py")),
        api = api,
        debug = debug,
    )

def format_model_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/model.py")),
        api = api,
        debug = debug,
    )

def format_ensemble_code(list_of_snippets, debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/codeg_ensemble.py")),
        import_snippets = "\n".join(f"from agents.codeg import {s}" for s in list_of_snippets),
        list_of_snippets = "[" + ", ".join([f"{s}.{api.name}" for s in list_of_snippets]) + "]",
        api = api,
        debug = debug,
    )

def format_pipeline_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/pipeline.py")),
        api = api,
        debug = debug,
    )

def format_evaluation_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/evaluation.py")),
        api = api,
        debug = debug,
    )

def format_train_model_code(debug=False, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("snippets/train_model.py")),
        project_name = kwargs['project_name'],
        api = api,
        debug = debug,
    )