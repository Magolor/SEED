from seed import *

<<api.api_def(with_kwargs=True)>>
    '''
<<indent(format_llmqa_single_prompt(examples=GetExamples("./"),**LoadJson("config.json")),1)>>
    '''
    config = LoadJson("config.json")
    instance = dict(<<api.asgs>>)
    examples = GetExamples("./", instance=instance)
    if 'advice' in kwargs:
        config['advice'] = kwargs['advice']
    prompt = format_llmqa_single_prompt(examples=examples, instance=instance, **config)
    response = LLM().q(messages = [
        {'role': 'user', 'content': prompt},
    ], post_processings = [parse_values])
    outputs = [(response['values'][o] if o in response['values'] else None) for o in <<[o['name'] for o in api.config['outputs']]>>]
    return tuple(outputs) if len(outputs) > 1 else outputs[0]