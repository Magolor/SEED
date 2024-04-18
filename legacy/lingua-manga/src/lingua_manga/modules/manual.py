from .default import *

@LinguaManga.register
class ManualModule(Module):
    __type__: str = 'module-manual'
    def __init__(self, name, code, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs); self.__type__ = self.__type__
        self.code = code
    
    def __compile__(self):
        return Cell(code=self.code)