from .utils import *
from .sandbox import *
import openai

class LLMCore(object):
    def __init__(self, backend="openai_gpt-4"):
        self.backend = backend
        if self.backend.startswith("openai_"):
            self.config = GetConfig('llms')['openai-api']
            openai.api_key = self.config['api_key']
            openai.organization = self.config['organization']
            self.model = backend.split('_')[-1]
        else:
            pass
        
    def Query(self, messages, retry_time=3, retry_gap=0.1, functions=list()):
        identifier = "|".join([self.backend, str(messages)] + ([str(functions)] if functions else []))
        response = retrieve_cache(identifier)
        if response is not None:
            return response
        while retry_time>0:
            try:
                if self.backend.startswith("openai_"):
                    if functions:
                        response = openai.ChatCompletion.create(
                            model = self.model,
                            messages = messages,
                            functions = functions,
                            temperature = 0,
                        ).choices[0]['message']
                    else:
                        response = openai.ChatCompletion.create(
                            model = self.model,
                            messages = messages,
                            temperature = 0,
                        ).choices[0]['message']
                    update_cache(identifier, response)
                    return response
                else:
                    raise NotImplementedError
            except KeyboardInterrupt as e:
                exit(0)
            except NotImplementedError as e:
                print(ERROR(e))
                exit(0)
            except Exception as e:
                print(ERROR(e))
                time.sleep(retry_gap)
                print(ERROR(f"Retrying..."))
                retry_time -= 1
        return None
    
llm = LLMCore()
def LLMQuery(messages):
    return llm.Query(messages)['content']

def LLMGC(messages, language="python"):
    response = LLMQuery(messages)
    assert (f"```{language}" in response), (response)
    assert ("```" in response.split(f"```{language}")[-1]), (response)
    prompt_doc = '"""\n'+('\n\n'.join([message['role']+":\n"+message['content'] for message in messages]))+'"""\n'
    return Cell(
        code = prompt_doc+(response.split(f"```{language}")[-1].split("```")[0].strip()),
        messages = messages + [{"role":"assistant", "content":response}]
    )

def LLMQueryExec(prompt, instances):
    instances_prompt = ""
    for i, inputs in enumerate(instances):
        instance_prompt = "\n".join([f"{k}={repr(v)}" for k,v in inputs.items()])
        instances_prompt += f"Instance #{i}:\n" + add_indent(instance_prompt)
    messages = [{"role":"user", "content":prompt.format(instance=instances_prompt)}]
    response = LLMQuery(messages)
    outputs = []
    for line in response.split("\n"):
        if line.startswith("Instance"):
            outputs.append(dict())
        if line.strip() and "=" in line:
            key, value = line.strip().split("=")
            outputs[-1][key] = eval(value)
    return outputs

def LLMToolExec(prompt, timeout=10):
    messages = [{"role":"user", "content":prompt}]; status = True
    while status:
        response = LLMQuery(messages)
        messages.append({"role":"assistant", "content":response})
        assert (response.startswith("Thought:")), (response)
        assert ("Action:" in response), (response)
        action = response.split("Action:")[-1].strip()
        code = Cell(
            "from apis import *\n"
            f"data = {action}\n"
        )
        print(response)
        observation, status = Exec(code)
        print(observation)
        messages.append({"role":"user", "content":f"Observation: {observation}\n"})
        assert (len(messages) <= timeout)
    print()
    return observation