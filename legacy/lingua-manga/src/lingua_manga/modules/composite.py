from .default import *

@LinguaManga.register
class CompositeModule(Module):
    __type__: str = 'module-composite'
    def __init__(self, name, modules, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        self.modules = modules
    
    def __compile__(self):
        code = ("\n".join([f"{module.api()}" for module in self.modules]) if self.modules else "...")+"\n"
        for module in self.modules:
            for key in module.outputs:
                if key not in self.outputs:
                    code += "del "+parameterize(key)+"\n"
        return Cell(f"def {self.name}(data):\n{add_indent(code)}")