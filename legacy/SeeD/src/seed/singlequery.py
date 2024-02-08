from .utils import *

LLMQUERY_SINGLE_CODE = """
from seed import *

class {name}_single_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def {api_self}:
        '''
{prompt}
        '''
        prompt = {prompt_repr}.format(instance_desc=format_instance(
            {api_copyargs}
        ))
        try:
            response = self.llm.Query([{{'role':'user', 'content':prompt}}])['content']
            self.responses.append(response)
        except Exception as e:
            print(e)
            self.responses.append(None)
        add_llm_count(type="llm", size=1, tokens=len(prompt))
        return True
"""
LLMQUERY_SINGLE_PROMPT_TEMPALTE = (
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "You are expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Now consider the following instance:\n"
    "{instance_desc}"
    "Please respond with the answer only. Please do not output any other responses or any explanations.\n"
)
EXAMPLE_PROMPT_TEMPLATE = (
    "Example #{idx}:\n"
    "Inputs:\n"
    "{inputs_desc}"
    "Output:\n"
    "{output_desc}"
    "{explanation_desc}"
)

def format_obj(obj):
    return obj if isinstance(obj, str) else repr(obj)

def format_inputs(inputs):
    return ("\n".join([f"- {k}: {v}" for k, v in inputs.items()]))+"\n"

def format_output(output):
    return f"- {output}" + "\n"

def format_example(idx, example):
    return EXAMPLE_PROMPT_TEMPLATE.format(idx=idx,
        inputs_desc=format_inputs({k:format_obj(v) for k,v in example["inputs"].items()}),
        output_desc=format_output(format_obj(example["output"])),
        explanation_desc=f"Explanation: {example['info']}" if "info" in example else "",
    )

def format_examples(examples):
    return "".join([format_example(idx, example) for idx, example in enumerate(examples)]) + ("\n" if examples else "")

def format_instance(**instance):
    return format_inputs({k:format_obj(v) for k,v in instance.items()})

def format_llmquery_single(desc, inputs, output, examples):
    return LLMQUERY_SINGLE_PROMPT_TEMPALTE.format(
        task_desc = desc + "\n",
        inputs_desc = format_inputs(inputs),
        output_desc = format_output(output),
        examples_desc = format_examples(examples),
        instance_desc = "{instance_desc}",
    ).replace('{','{{').replace('}','}}').replace('{{instance_desc}}', '{instance_desc}')

def format_llmquery_single_code(cell):
    prompt = format_llmquery_single(
        desc = cell.desc,
        inputs = cell.inputs,
        output = cell.output,
        examples = cell.examples[:10],
    )
    return LLMQUERY_SINGLE_CODE.format(
        name = cell.name,
        prompt = add_indent(prompt.replace("{instance_desc}", "{instance_desc}\n"), indent=2),
        prompt_repr = repr(prompt),
        api_self = cell.api_self(),
        api_copyargs = cell.api_copyargs(),
    )