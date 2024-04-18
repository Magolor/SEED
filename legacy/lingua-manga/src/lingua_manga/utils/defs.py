from .utils import *

class Data(MemberDict):
    def save(self, path, indent=None):
        CreateFile(AsFormat(path, "json")); SaveJson(dict(self), AsFormat(path, "json"), indent=indent)
    
    @classmethod
    def load(cls, path):
        return cls(LoadJson(AsFormat(path, "json")))

class ModuleManager:
    def __init__(self):
        self.types = {}

    def get(self, type):
        return self.types[type]

    @property
    def register(self):
        def r(m):
            self.types[m.__type__] = m
            return m
        return r
    
    def load(self, path):
        data = Data.load(path)
        type = self.get(data.__type__)
        return type(**dict(data))

LinguaManga = ModuleManager()
