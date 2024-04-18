from .utils import *

LLMQUERY_TOOLS_CODE = """
from seed import *

class {name}_tools_class(object):
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
            messages = [{{'role':'user', 'content':prompt}}]; status = True
            while status:
                llm_response = self.llm.Query(messages)['content']
                messages.append({{"role":"assistant", "content":llm_response}})
                assert (llm_response.startswith("Thought:")), (llm_response)
                assert ("Action:" in llm_response), (llm_response)
                thought = llm_response.split("Thought:")[-1].split("Action:")[0].strip()
                action = llm_response.split("Action:")[-1].strip()
                print(thought)
                print(action)
                env = Environment()
                code = ScriptCell(
                    "from apis import *\\n"
                    f"{{action}}\\n"
                )
                env.add(code)
                observation, status = env.reset()['response']
                print(observation)
                if status:
                    messages.append({{"role":"user", "content":f"Observation: {{observation}}\\n"}})
            print(observation)
            self.responses.append(observation)
            add_llm_count(type="llm", size=len(messages)//2, tokens=len(prompt))
        except Exception as e:
            print(e)
            self.responses.append(None)
        return True
"""
LLMQUERY_TOOLS_PROMPT_TEMPALTE = (
    "{task_desc}"
    "Please do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:\n"
    "{tools_desc}"
    "Interaction Examples:\n"
    "{examples_desc}"
    "Now consider the following instance:\n"
    "{instance_desc}"
    "Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.\n"
)
EXAMPLE_PROMPT_TEMPLATE = (
    "Example #{idx}:\n"
    "Inputs:\n"
    "{inputs_desc}"
    "Interactions:\n"
    "{interactions_desc}"
    "{explanation_desc}"
)

def format_obj(obj):
    return obj if isinstance(obj, str) else repr(obj)

def format_inputs(inputs):
    return ("\n".join([f"- {k}: {v}" for k, v in inputs.items()]))+"\n"

def format_output(output):
    return f"- {output}" + "\n"

def format_toolsquery(desc, tools, examples_desc):
    return LLMQUERY_TOOLS_PROMPT_TEMPALTE.format(
        task_desc = desc + "\n",
        tools_desc = format_inputs(tools),
        examples_desc = examples_desc,
        instance_desc = "{instance_desc}",
    ).replace('{','{{').replace('}','}}').replace('{{instance_desc}}', '{instance_desc}')

def format_interaction_example(idx, example):
    return EXAMPLE_PROMPT_TEMPLATE.format(idx=idx,
        inputs_desc=format_inputs({k:format_obj(v) for k,v in example["inputs"].items()}),
        interactions_desc=("\n".join(example["interaction"])+"\n") if "interaction" in example else "",
        explanation_desc=f"Explanation: {example['info']}" if "info" in example else "",
    )

def format_interaction_examples(examples):
    return "".join([format_interaction_example(idx, example) for idx, example in enumerate(examples)]) + ("\n" if examples else "")

def format_llmquery_tools_code(cell):
    prompt = format_toolsquery(
        desc = cell.desc,
        tools = cell.query_tools,
        examples_desc = format_interaction_examples(cell.examples[:10]),
    )
    return LLMQUERY_TOOLS_CODE.format(
        name = cell.name,
        prompt = add_indent(prompt.replace("{instance_desc}", "{instance_desc}\n"), indent=2),
        prompt_repr = repr(prompt),
        api_self = cell.api_self(),
        api_copyargs = cell.api_copyargs(),
    )
