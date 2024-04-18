from ..utils import *
from .snippets import *

def format_parameters(**kwargs):
    return '\n'.join([safe_format(
        LoadText(find_resource("prompts/default_par.txt")),
        key = key,
        value = '?' if value is ... else value,
    ) for key, value in kwargs.items()])+'\n'

# name, desc
def format_description(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/default_desc.txt")),
        name = kwargs['name'],
        desc = kwargs['desc'],
    )

def format_descriptions(l):
    return '\n'.join(format_description(**x) for x in l)+'\n'

# inputs, outputs, [explanation]
def format_example(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/default_exp.txt")),
        idx = kwargs['idx'],
        inputs_desc = format_parameters(**kwargs['inputs']),
        expected_desc = format_parameters(**kwargs['outputs']),
        explanation = kwargs['explanation'] if 'explanation' in kwargs else '',
    )

def format_examples(l):
    return '\n'.join(format_example(**(x | {'idx':i})) for i,x in enumerate(l,1))+'\n'

# inputs, outputs, responses/error, [explanation]
def format_error(**kwargs):
    if 'responses' in kwargs:
        kwargs['error'] = format_parameters(**kwargs['responses'])
    assert ('error' in kwargs), "Error message is required!"
    return safe_format(
        LoadText(find_resource("prompts/default_err.txt")),
        inputs_desc = format_parameters(**kwargs['inputs']),
        expected_desc = format_parameters(**kwargs['outputs']),
        outputs_desc = kwargs['error'],
        explanation = kwargs['explanation'] if 'explanation' in kwargs else '',
    )

def format_errors(l):
    return '\n'.join(format_error(x | {'idx':i}) for i,x in enumerate(l,1))+'\n'

# task_desc, inputs, outputs, examples
def format_profile(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/default_prof.txt")),
        task_desc = kwargs['task_desc'],
        inputs_desc = format_descriptions(kwargs['inputs']),
        outputs_desc = format_descriptions(kwargs['outputs']),
        examples_desc = format_examples(kwargs['examples']) if 'examples' in kwargs else '',
    )

# task_desc, inputs, outputs, examples
def format_llmqa_single_prompt(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/llmqa_single.txt")),
        task_profile = format_profile(**kwargs),
        examples_desc = format_examples(kwargs['examples']) if 'examples' in kwargs else '',
        instance_desc = format_instance(kwargs['instance']) if 'instance' in kwargs else '',
    )

def format_instance(instance):
    return safe_format(
        LoadText(find_resource("prompts/default_inst.txt")),
        inputs_desc = format_parameters(**instance),
    )

def format_codeg_template(**kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("prompts/codeg_template.txt")),
        api = api,
    )

def format_advg_prompt(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/codeg_advg.txt")),
        task_profile = format_profile(**kwargs),
    )

def format_advsg_prompt(**kwargs):
    return safe_format(
        LoadText(find_resource("prompts/codeg_advsg.txt")),
        task_profile = format_profile(**kwargs),
        codeg_branches_count = kwargs['codeg_branches_count'],
    )

def format_tempg_prompt(**kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("prompts/codeg_tempg.txt")),
        api = api,
        task_profile = format_profile(**kwargs),
        advice = kwargs['advice'] if 'advice' in kwargs else '',
        examples_desc = format_examples(kwargs['examples']) if 'examples' in kwargs else '',
    )

def format_codeg_prompt(advice='', **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("prompts/codeg.txt")),
        api = api,
        task_profile = format_profile(**kwargs),
        template_desc = format_codeg_template(**kwargs),
        advice = advice,
    )

def format_fallback_prompt(**kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("prompts/codeg_fallback.txt")),
        api = api,
        task_profile = format_profile(**kwargs),
        code = kwargs['code'] if 'code' in kwargs else '',
    )

def format_codeg_expfix_prompt(code, example, **kwargs):
    api = API(**kwargs)
    return safe_format(
        LoadText(find_resource("prompts/codeg_expfix.txt")),
        api = api,
        task_profile = format_profile(**kwargs),
        example_desc = format_error(**example),
        template_desc = format_codeg_template(**kwargs),
        code = code,
    )