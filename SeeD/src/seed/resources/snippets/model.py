from seed import *

model_agent = None
def get_model_agent():
    global model_agent
    if model_agent is None: model_agent = ModelAgent(**LoadJson("config.json"))
    return model_agent

<<api.api_def(with_kwargs=True)>>
    instance = dict(<<api.asgs>>)
    model_agent = get_model_agent()
    pred, conf = model_agent.run(instance)
    thres = model_agent.get_confidence_threshold()
    print("model", pred, conf, thres)
    return pred if conf >= thres else None