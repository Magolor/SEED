
from seed import *

class data_imputation_simul_class(object):
    def __init__(self, module_path, simul_type, simul_threshold, num_classes, custom=None):
        self.simul = Simul(module_path, simul_type, num_classes, custom=custom)
        self.confidence = simul_threshold
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
        value, confidence = self.simul.get_simul(key)
        print(key, value, confidence, self.confidence)
        if (value is not None) and (confidence >= self.confidence):
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
        self.simul.add_simul(key, value)
