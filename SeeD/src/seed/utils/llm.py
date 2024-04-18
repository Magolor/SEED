from .utils import *

class LLM:
    def __init__(self):
        self.model = get_config('llm_config')['model']
        self.temperature = get_config('llm_config')['temperature']
        self.sleep = get_config('llm_config')['sleep']
        self.retry = get_config('llm_config')['retry']
        openai.api_key = get_config('llm_config')['api_key']
        assert openai.api_key!='<OPENAI_API_KEY>', "Please set your OpenAI API key during installation."
        openai.organization = get_config('llm_config')['organization']
        assert openai.organization!='<OPENAI_ORG>', "Please set your OpenAI organization key during installation."
        try:
            self.client = OpenAI(api_key=openai.api_key, organization=openai.organization)
        except Exception as e:
            print(ERROR(e))
            exit(0)
    
    def q(self, messages, functions=list(), post_processings=list()):
        identifier = repr(messages) + '|' + repr(functions)
        cached = get_exact_cache(identifier)
        if cached is not None:
            response = cached
        else:
            for t in range(self.retry):
                try:
                    completion = self.client.chat.completions.create(
                        model = self.model,
                        messages = messages,
                        functions = functions if functions else NOT_GIVEN,
                        temperature = self.temperature,
                    )
                    response = {
                        'status': True,
                        'text': completion.choices[0].message.content,
                        'tools': completion.choices[0].message.tool_calls if functions and completion.choices[0].tool_calls else list(),
                        # 'response': completion,
                        'msg': None,
                    }
                except Exception as e:
                    response = {
                        'status': False,
                        'text': None,
                        # 'response': None,
                        'msg': str(e),
                    }
                time.sleep(self.sleep)
                if response['text'] is not None:
                    break
            if response['text'] is not None:
                add_exact_cache(identifier, response)
        for p in post_processings:
            response = p(response)
        return response

def parse_code(response, language='Python'):
    ordered_keys = [
        f'```{language}',
        f'```{language.lower()}',
        f'```'
    ]
    code = None
    for key in ordered_keys:
        if response['status'] and response['text'] and (key in response['text']):
            code = response['text'].split(key)[1].split('```')[0].strip()
            break
    return response | {'code': code}

def parse_multiple_outputs(response, outputs=['Thought', 'Action']):
    if (not response['status']) or (not response['text']):
        return response | {output.lower(): None for output in outputs}
    data = {}; text = response['text']
    for key in reversed(outputs):
        if key+':' in text:
            remain, value = text.rsplit(key+':', 1)
            data[key.lower()] = value.strip()
            text = remain
        else:
            data[key.lower()] = None
    return response | data

# def parse_cot(response, thought='Thought', action='Action'):
#     thought, action = None, None
#     if response['status'] and response['text'] and (thought in response['text']) and (action in response['text']):
#         thought = response['text'].split(thought)[1].split(action)[0].strip()
#         action = response['text'].split(action)[1].strip()
#     elif response['status'] and response['text'] and (thought in response['text']):
#         thought = response['text'].split(thought)[1].strip()
#     return response | {'thought': thought, 'action': action}
def parse_cot(response):
    return parse_multiple_outputs(response, outputs=['Thought', 'Action'])

def parse_ideas(response, limit=3):
    return parse_multiple_outputs(response, outputs=[f'Idea {i}' for i in range(1, limit+1)])

def parse_values(response):
    key_value_pairs = [l.split(' = ',1) for l in response['text'].split('\n') if len(l.split(' = ',1))==2] if response['status'] else list()
    return response | {'values': {k:safe_eval(v) for k,v in key_value_pairs} if response['status'] else None}