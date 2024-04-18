from .utils import *
from .modules import Module, ManualModule, LLMGCModule, LLMToolModule, LLMQueryModule, LLMGCProModule
from .cacher import KVCache
from .batcher import SimilarBatch, DiverseBatch, ClosestBatch, FarthestBatch
from .simulator import LMSimulator, CLSSimulator
from .validator import NoneValidator, ExampleVerifier, FuzzyExampleVerifier, AdversarialVerifier

def CreateProject(name, clear=False):
    projects = GetConfig("paths")['projects_path']
    root = pjoin(projects, name); CreateFolder(root)
    data = pjoin(root, "data"); CreateFolder(data)
    apis = pjoin(root, "apis"); CreateFolder(apis)
    cache = pjoin(root, "cache"); ClearFolder(cache) if clear else CreateFolder(cache)
    workspace = pjoin(root, "workspace"); ClearFolder(workspace) if clear else CreateFolder(workspace)
    modules = pjoin(root, "modules"); ClearFolder(modules) if clear else CreateFolder(modules)
    simuls = pjoin(root, "simuls"); ClearFolder(simuls) if clear else CreateFolder(simuls)
    KVCache(path=cache, init="sentence-transformers/all-MiniLM-L12-v2")
    
def AddModule(project, module):
    projects = GetConfig("paths")['projects_path']
    root = pjoin(projects, project); CreateFolder(root)
    modules = pjoin(root, "modules"); CreateFolder(modules)
    module_path = pjoin(modules, module.name); CreateFolder(module_path)
    module.save(pjoin(module_path, "config.json"), indent=4)

def DeleteModule(project, module_name):
    projects = GetConfig("paths")['projects_path']
    root = pjoin(projects, project); CreateFolder(root)
    modules = pjoin(root, "modules"); CreateFolder(modules)
    module_path = pjoin(modules, module_name); Delete(module_path)

def CompileModule(project, module_name, validator, prev_cells=list()):
    projects = GetConfig("paths")['projects_path']
    root = pjoin(projects, project); CreateFolder(root)
    modules = pjoin(root, "modules"); CreateFolder(modules)
    module_path = pjoin(modules, module_name); assert ExistFolder(module_path)
    module = LinguaManga.load(pjoin(module_path, "config.json"))
    module.validator = validator
    module.prev_cells = prev_cells
    module.compile()
    module.validator = None
    module.prev_cells = None
    module.save(pjoin(module_path, "config.json"), indent=4)
    return module.__executable__

def CompileProject(project, validator=NoneValidator()):
    projects = GetConfig("paths")['projects_path']
    root = pjoin(projects, project); CreateFolder(root)
    modules = pjoin(root, "modules"); CreateFolder(modules)
    executables = []
    executables.append(
        Cell(code = "from lingua_manga import *")
    )
    for module_name in ListFolders(modules, ordered=True):
        executables.append(CompileModule(project, module_name, validator, prev_cells=executables))
    compiled = pjoin(root, AsFormat(project, "py"))
    with open(compiled, "w") as f:
        f.write("# %%\n" + ("# %%\n".join([e.code+"\n\n" for e in executables])))
        f.write("# %%\n")