from .utils import *

LLMQUERY_BATCH_CODE = """
from seed import *

class {name}_batch_class(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.buffer = list()
        self.responses = list()
        
    def {api_self}:
        self.buffer.append(Data(
            {api_copyargs}
        ))
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False
    
    def flush(self):
        prompt = {prompt_repr}.format(instances_desc=format_instances(self.buffer))
        try:
            response = self.llm.Query([{{'role':'user', 'content':prompt}}])['content']
            lines = [line.strip() for line in response.strip().split("\\n") if line.strip()]
            assert (len(lines) == len(self.buffer)), "Incorrect Responses!"
            values = []
            for idx, line in enumerate(lines, 1):
                assert (line.startswith(f"Output #{{idx}}: ")), "Incorrect Responses!"
                value_repr = line.split(f"Output #{{idx}}: ")[-1].strip()
                try:
                    value = eval(value_repr)
                except:
                    value = value_repr
                values.append(value)
            self.responses.extend(values)
        except:
            self.responses.extend([None for _ in self.buffer])
        if len(self.buffer)>0:
            add_llm_count(type="llm", size=len(self.buffer), tokens=len(prompt))
        self.buffer = list()
"""
LLMQUERY_BATCH_PROMPT_TEMPALTE = (
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "You are expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Now consider the following instances:\n"
    "{instances_desc}"
    "Please respond with the answer only, one line for each instance. Please do not output any other responses or any explanations.\n"
    "Each response should start with \"Output #<index>: \". For example:\n"
    "Output #1: ...\n"
    "Output #2: ...\n"
    "...\n"
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

def format_instances(instances):
    return "".join([f"Instance #{idx}:\n"+format_instance(**instance) for idx, instance in enumerate(instances,1)]) + ("\n" if instances else "")

def format_llmquery_batch(desc, inputs, output, examples):
    return LLMQUERY_BATCH_PROMPT_TEMPALTE.format(
        task_desc = desc + "\n",
        inputs_desc = format_inputs(inputs),
        output_desc = format_output(output),
        examples_desc = format_examples(examples),
        instances_desc = "{instances_desc}",
    ).replace('{','{{').replace('}','}}').replace('{{instances_desc}}', '{instances_desc}')

def format_llmquery_batch_code(cell):
    prompt = format_llmquery_batch(
        desc = cell.desc,
        inputs = cell.inputs,
        output = cell.output,
        examples = cell.examples[:10],
    )
    return LLMQUERY_BATCH_CODE.format(
        name = cell.name,
        prompt_repr = repr(prompt),
        api_self = cell.api_self(),
        api_copyargs = cell.api_copyargs(),
    )