from ..templates import *

class Agent:
    def __init__(self, **kwargs):
        self.config = LoadJson(pjoin(kwargs['project_path'], "config.json"))
        self.path = pjoin(self.config['project_path'], "agents", "empty")
    
    def compile(self):
        CreateFolder(self.path); CreateFile(pjoin(self.path,'__init__.py'))
        with open(pjoin(self.path,'__init__.py'), 'w') as f:
            f.write(format_default_code(**self.config))