from .utils import *
from .sandbox import get_globals, get_seed_globals

CUSTOM_FORMATTER = ["<<", ">>"]
SEED_GLOBALS = None
def safe_eval_pattern(s, globals={}, locals={}):
    try:
        return eval(s, globals, locals)
    except Exception as e:
        return CUSTOM_FORMATTER[0] + s + CUSTOM_FORMATTER[1]
def safe_format(string, **kwargs):
    locals = {k:v for k, v in kwargs.copy().items() if v!=...}
    pattern = CUSTOM_FORMATTER[0] + r"(.*?)" + CUSTOM_FORMATTER[1]
    expressions = re.findall(pattern, string)
    formatted = string
    global SEED_GLOBALS
    if SEED_GLOBALS is None:
        SEED_GLOBALS = get_seed_globals()
    for expr in expressions:
        try:
            result = safe_eval_pattern(expr, globals=SEED_GLOBALS, locals=locals)
            if result != ...:
                formatted = formatted.replace(CUSTOM_FORMATTER[0] + expr + CUSTOM_FORMATTER[1], str(result).rstrip())
        except Exception as e:
            print(e)
            pass
    return formatted