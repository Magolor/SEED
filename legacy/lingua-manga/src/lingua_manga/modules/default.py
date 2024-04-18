from ..utils import *

@LinguaManga.register
class Module(Data):
    __type__: str = 'module-default'
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs); self.__type__ = self.__type__
        self.name = name
        self.__info__ = Data()
        self.__executable__ = None
        self.__version__ = 1
        self.__compiled_version__ = 0
        if self.desc is None: self.desc = ""
        if self.inputs is None: self.inputs = list()
        if self.outputs is None: self.outputs = list()
        if self.examples is None: self.examples = list()
        if self.optimizers is None: self.optimizers = list()
        
    def __compile__(self):
        return Cell(f"def {self.api()}:\n{add_indent('raise NotImplementedError')}")
    
    def compile(self, *args, **kwargs):
        if self.__compiled_version__ < self.__version__:
            self.__executable__ = self.__compile__(*args, **kwargs)
            self.__compiled_version__ = self.__version__
        for optimizer in self.optimizers:
            optimizer.optimize(self)
        return self.__executable__
    
    def arguments(self):
        return f"({', '.join([i for i in self.inputs])})"

    def assignments(self, **kwargs):
        return f"({', '.join([f'{k}={kwargs[k]}' for k in self.inputs])})"
    
    def api(self):
        return f"{self.name}" + self.arguments()
    
    def api_call(self, **kwargs):
        return f"{self.name}" + self.assignments(**kwargs)
    
    def batch_api(self):
        return f"{self.name}_batch(instances)"
    
    def optimized_api(self):
        return f"{self.name}_optimized(instances,cacher=True,simulator=True,batching=8)"