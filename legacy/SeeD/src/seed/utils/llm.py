from .defs import *

class LLMCore(object):
    def __init__(self, backend="openai_gpt-4"):
        self.backend = backend
        if self.backend.startswith("openai_"):
            self.config = get_config('openai-api')
            openai.api_key = self.config['api_key']
            openai.organization = self.config['organization']
            self.model = backend.split('_')[-1]
        else:
            pass
        
    def Query(self, messages, functions=list(), retry_time=3, retry_gap=0.1):
        identifier = "|".join([self.backend, str(messages)] + ([str(functions)] if functions else []))
        response = get_exact_cache(identifier)
        if response is not None:
            return response
        while retry_time>0:
            try:
                if self.backend.startswith("openai_"):
                    if functions:
                        response = openai.chat.completions.create(
                            model = self.model,
                            messages = messages,
                            functions = functions,
                            temperature = 0,
                        ).choices[0].message
                    else:
                        response = openai.chat.completions.create(
                            model = self.model,
                            messages = messages,
                            temperature = 0,
                        ).choices[0].message
                    response = {'role':response.role, 'content':response.content, 'function_call':response.function_call, 'tool_calls':response.tool_calls}
                    add_exact_cache(identifier, response)
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