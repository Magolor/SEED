from .utils import *

class Data(MemberDict):
    def save(self, path, indent=None):
        CreateFile(AsFormat(path, "json")); SaveJson(dict(self), AsFormat(path, "json"), indent=indent)
    
    @classmethod
    def load(cls, path):
        return cls(LoadJson(AsFormat(path, "json")))
    
    def clone(self):
        return self.__class__(deepcopy(dict(self)))

class Cell(Data):
    def __init__(self, name, inputs, output, code="", examples=list(), import_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.inputs = inputs
        self.output = output
        self.code = code
        self.examples = examples
        self.import_path = import_path
    
    def api_def(self, _suffix=""):
        return self.name + _suffix + "(" + ', '.join([key for key in self.inputs]) + ")"
    
    def api_call(self, _suffix="", **kwargs):
        return self.name + _suffix + "(" + ', '.join([key+"="+kwargs[key] for key in self.inputs]) + ")"
    
    def api_kwargs(self, _suffix=""):
        return self.name + _suffix + "(**kwargs)"
    
    def api_copyargs(self):
        return ',\n'.join([key+"="+key for key in self.inputs])
    
    def api_self(self):
        return "__call__" + "(" + ', '.join(["self"] + [key for key in self.inputs]) + ")"
    
    def api_update(self):
        return "update" + "(" + ', '.join(["self"] + [key for key in self.inputs] + ["value"]) + ")"
    
    def api_import(self, _suffix=""):
        return f"from {self.import_path} import {self.name + _suffix}"

class ScriptCell(Data):
    def __init__(self, code="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code

class Environment(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shell = None; self.cells = list()
    
    def save(self, path, indent=None):
        self.shell = None; super().save(path, indent=indent)

    def execute(self, code):
        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                response = self.shell.run_cell(code)
            if response.success:
                return {'status':True, 'response':self.shell.user_ns['_'], 'code':code, 'msg':None}
            else:
                return {'status':False, 'response':self.shell.user_ns['_'], 'code':code, 'msg':str(response.error_in_exec)}
        except Exception as e:
            return {'status':False, 'response':self.shell.user_ns['_'], 'code':code, 'msg':str(type(e))+' '+str(e)}

    def reset(self):
        self.shell = embed.InteractiveShellEmbed()
        for cell in self.cells:
            response = self.shell.run_cell(cell.code)
            if not response.success:
                return {'status':False, 'response':None, 'cell':cell, 'msg':response['msg']}
        return {'status':True, 'response':self.shell.user_ns['_'], 'cell':None, 'msg':None}
    
    def add(self, cell):
        self.cells.append(cell)
    
    def get(self, key="_"):
        return self.shell.user_ns[key] if self.shell else None
    
    def export(self, path):
        with open(AsFormat(path, "py"), "w") as f:
            for cell in self.cells:
                f.write("# %%\n")
                f.write(cell.code)
                f.write("\n\n")
            f.write("# %%\n")
    
    def exit(self):
        self.shell.execute("exit")