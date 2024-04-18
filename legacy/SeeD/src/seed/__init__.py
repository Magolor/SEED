from .utils import *
from .singlequery import *
from .batchquery import *
from .toolsquery import *
from .cache import *
from .codegen import *
from .simul import *
from .integrated import *

def get_seed_args():
    args = HeavenArguments.from_parser([
        SwitchArgumentDescriptor("use_cache",           short="C",                              help="Whether to use cache.",),
        # FloatArgumentDescriptor("cache_threshold",      short="r",              default=0.3,    help="A portion r of the embeddings are considered matched.",),
        FloatArgumentDescriptor("cache_threshold",      short="d",              default=0.3,    help="Two unit embeddings are matched if their l2-distance is no greater than 2d.",),

        SwitchArgumentDescriptor("use_codegen",         short="G",                              help="Whether to use code generation.",),
        SwitchArgumentDescriptor("use_ensemble",        short="e",                              help="Whether to use code ensemble.",),
        IntArgumentDescriptor("timeout",                short="T",              default=5,      help="The max number of validation retries.",),
        SwitchArgumentDescriptor("use_logical_eval",    short="eval-logical",                   help="Whether to use logical correctness evaluation.",),
        SwitchArgumentDescriptor("use_example_eval",    short="eval-example",                   help="Whether to use example-based verfication.",),
        SwitchArgumentDescriptor("use_testgen_eval",    short="eval-testgen",                   help="Whether to use test case generation.",),

        SwitchArgumentDescriptor("use_simul",           short="S",                              help="Whether to use simulation.",),
        StrArgumentDescriptor("custom_checkpoint",      short="ckpt",                           help="Whether to use a custom checkpoint.",),
        LiteralArgumentDescriptor("simul_type",         short="m",              default="clslm",help="The simulation type.", choices = ["seqlm", "clslm", "reglm"],),
        IntArgumentDescriptor("simul_num_classes",      short="n",              default=2,      help="For 'clslm' type, the number of classes.",),
        FloatArgumentDescriptor("simul_threshold",      short="c",              default=0.6,    help="Simulation is used if confidence is no less than c.",),
        
        SwitchArgumentDescriptor("use_tools",           short="X",                              help="Whether to use tools.",),

        SwitchArgumentDescriptor("use_batch",           short="B",                              help="Whether to use query batching.",),
        IntArgumentDescriptor("batch_size",             short="b",              default=32,     help="The number of queries in one batch.",),
        SwitchArgumentDescriptor("no_query",            short="nq",                             help="Whether NOT to use llm query.",),
        
        LiteralArgumentDescriptor("reorder",            short="o",              default="RND",  help="The ordering strategy.", choices = ["RND", "SIM", "DIV", "CLS", "FAR"],),
        IntArgumentDescriptor("random_seed",            short="seed",           default=42),
        
        StrArgumentDescriptor("identifier",             short="i",             default="base"),
    ])
    return args

def CreateProject(name, clear=False):
    projects = get_config('projects_path')
    root = pjoin(projects, name); CreateFolder(root)
    data = pjoin(root, "data"); CreateFolder(data)
    modules = pjoin(root, "modules"); ClearFolder(modules,rm=True) if clear else CreateFolder(modules)
    CreateFile(pjoin(modules, "__init__.py"))
    
def AddModule(project, name, desc, inputs, output, examples, code_tools, query_tools, args):
    projects = get_config('projects_path')
    root = pjoin(projects, project)
    modules = pjoin(root, "modules")
    module_root = pjoin(modules, name); CreateFolder(module_root)
    cache_root = pjoin(module_root, "cache"); CreateFolder(cache_root)
    
    with open(pjoin(module_root, "__init__.py"), "w") as f:
        f.write("from seed import *\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
    cell.code = format_llmquery_single_code(cell)
    with open(pjoin(module_root, f"{name}_singlequery.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_singlequery"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_single_class") + "\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
    cell.code = format_llmquery_batch_code(cell)
    with open(pjoin(module_root, f"{name}_batchquery.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_batchquery"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_batch_class") + "\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples, query_tools = query_tools,)
    cell.code = format_llmquery_tools_code(cell)
    with open(pjoin(module_root, f"{name}_toolsquery.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_toolsquery"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_tools_class") + "\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
    cell.code = format_cache_code(cell)
    with open(pjoin(module_root, f"{name}_cache.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_cache"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_cache_class") + "\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
    cell.code = format_simul_code(cell)
    with open(pjoin(module_root, f"{name}_simul.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_simul"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_simul_class") + "\n")
    
    if args.use_codegen:
        cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples, code_tools = code_tools,)
        code_cells = code_generation(cell, args=args)
        for code_cell in code_cells:
            cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
            cell.code = code_cell.doc + "\n" + code_cell.code
            with open(pjoin(module_root, f"{name}_code_V{code_cell.version}.py"), "w") as f:
                f.write(cell.code)
        
        cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
        cell.code = format_ensemble_code(cell, code_cells)
        with open(pjoin(module_root, f"{name}_ensembled.py"), "w") as f:
            f.write(cell.code)
        cell.import_path = f".{name}_ensembled"
        with open(pjoin(module_root, "__init__.py"), "a") as f:
            f.write(cell.api_import() + "\n")
    else:
        cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
        cell.code = "def {api_def}:\n    return None\n".format(api_def=cell.api_def())
        with open(pjoin(module_root, f"{name}_ensembled.py"), "w") as f:
            f.write(cell.code)
        cell.import_path = f".{name}_ensembled"
        with open(pjoin(module_root, "__init__.py"), "a") as f:
            f.write(cell.api_import() + "\n")
    
    cell = Cell(name = name, desc = desc, inputs = inputs, output = output, examples = examples,)
    cell.code = format_integrated_code(cell)
    with open(pjoin(module_root, f"{name}_integrated.py"), "w") as f:
        f.write(cell.code)
    cell.import_path = f".{name}_integrated"
    with open(pjoin(module_root, "__init__.py"), "a") as f:
        f.write(cell.api_import(_suffix="_integrated_class") + "\n")
    
    with open(pjoin(modules, "__init__.py"), "a") as f:
        f.write(f"from .{name} import *\n")
