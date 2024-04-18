from ..templates import *
from .agent import Agent

class LLMQAAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = pjoin(self.config['project_path'], "agents", "llmqa")

    def compile(self):
        CreateFolder(self.path); CreateFile(pjoin(self.path,'__init__.py'))
        with open(pjoin(self.path,'__init__.py'), 'w') as f:
            f.write(format_llmqa_single_code(**self.config))