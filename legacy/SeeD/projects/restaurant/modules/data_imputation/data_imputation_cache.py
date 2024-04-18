
from seed import *

class data_imputation_cache_class(object):
    def __init__(self, module_path, cache_threshold):
        self.threshold = cache_threshold
        self.cache = Cache(module_path)
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, name, addr, phone, type):
        key = serialize(
            task="data_imputation",
            name=name,
addr=addr,
phone=phone,
type=type
        )
        found, value, dist = self.cache.get_cache(key)
        if (found is not None) and (dist <= 2*self.threshold):
            self.responses.append(value)
        else:
            self.responses.append(None)
        return True
    
    def update(self, name, addr, phone, type, value):
        key = serialize(
            task="data_imputation",
            name=name,
addr=addr,
phone=phone,
type=type
        )
        self.cache.add_cache(key, value)
