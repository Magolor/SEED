from .defs import *

class Cell(Data):
    __type__: str = 'cell'
    def __init__(self, code="", *args, **kwargs):
        super().__init__(*args, **kwargs); self.__type__ = self.__type__
        self.code = code
    
    def execute_in(self, shell):
        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                response = shell.run_cell(self.code)
            if response.success:
                return {'status':True, 'response':shell.user_ns['_'], 'cell':self, 'msg':None}
            else:
                return {'status':False, 'response':None, 'cell':self, 'msg':str(response.error_in_exec)}
        except Exception as e:
            print(ERROR(e))
            return {'status':False, 'response':None, 'cell':self, 'msg':str(e)}
        
class Environment(Data):
    __type__: str = 'env'
    def __init__(self, env_cells=list(), *args, **kwargs):
        super().__init__(*args, **kwargs); self.__type__ = self.__type__
        self.env_cells = env_cells
        
    def reset(self):
        self.shell = embed.InteractiveShellEmbed()
        for cell in self.env_cells:
            response = cell.execute_in(self.shell)
            if not response['status']:
                return {'status':False, 'response':None, 'cell':cell, 'msg':response['msg']}
        return {'status':True, 'response':None, 'cell':None, 'msg':None}

    def execute(self, cell=None, reset=True):
        if reset:
            setup_response = self.reset()
            if not setup_response['status']:
                return setup_response
        if cell is not None:
            return cell.execute_in(self.shell)
        else:
            return setup_response
    
    def batch_execute(self, cells):
        return [self.execute(cell, reset=True) for cell in cells]
    
    @property
    def data(self, param="data"):
        return self.shell.user_ns[param]
    
    def export(self, path, main=None):
        with open(AsFormat(path, "py"), "w") as f:
            cells = self.env_cells + ([main] if main is not None else [])
            for cell in cells:
                f.write("# %%\n")
                f.write(cell.code)
                f.write("\n\n")
            f.write("# %%\n")
    
    def exit(self):
        self.execute(Cell(code="exit"), reset=False)

def Exec(cell):
    env = Environment()
    env.execute(cell=cell)
    data = env.data
    env.exit()
    return data