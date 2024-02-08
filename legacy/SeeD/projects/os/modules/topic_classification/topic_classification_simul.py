
from seed import *

class topic_classification_simul_class(object):
    def __init__(self, module_path, simul_type, simul_threshold, num_classes):
        self.simul = Simul(module_path, simul_type, num_classes)
        self.confidence = simul_threshold
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, text):
        key = serialize(
            task="topic_classification",
            text=text
        )
        value, confidence = self.simul.get_simul(key)
        print(key, value, confidence, self.confidence)
        if (value is not None) and (confidence >= self.confidence):
            self.responses.append(value)
        else:
            self.responses.append(None)
        return True
    
    def update(self, text, value):
        key = serialize(
            task="topic_classification",
            text=text
        )
        self.simul.add_simul(key, value)
