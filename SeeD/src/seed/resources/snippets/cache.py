from seed import *

cache_agent = None
def get_cache_agent():
    global cache_agent
    if cache_agent is None: cache_agent = CacheAgent(**LoadJson("config.json"))
    return cache_agent

<<api.api_def(with_kwargs=True)>>
    instance = dict(<<api.asgs>>)
    cache_agent = get_cache_agent()
    pred, conf = cache_agent.run(instance)
    thres = cache_agent.get_confidence_threshold()
    print("cache", pred, conf, thres)
    return pred if conf >= thres else None