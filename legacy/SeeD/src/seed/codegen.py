from .utils import *

CODE_ENSEMBLE_CODE = """
from seed import *

FUNCTIONS = []
{code_imports}

def {api_def}:
    responses = []
    for function in FUNCTIONS:
        try:
            responses.append(
                function(
                    {api_copyargs}
                )
            )
        except Exception as e:
            print(e)
            responses.append(None)

    # If all responses are None, return None
    if responses == [None for _ in responses]:
        return None
    
    # If a majority of respones exists, return the majority
    hist = sorted(Counter(responses).items(),key=lambda x:x[1],reverse=True)
    if (len(hist)==1) or (len(hist)>1 and hist[0][1]>hist[1][1]):
        return hist[0][0]
    
    # If a majority of respones does not exist, return the first non-None response
    for response in responses:
        if response is not None:
            return response
"""

ADVICE_PROMPT_TEMPLATE = (
    "Please help me with the following Python programming task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "{tools_desc}"
    "Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, please return `None` to abstain instead of returning an incorrect guess.\n"
    "The generated function should be robust, instead of only passing the provided examples. You are allowed use global variables to keep track of new incoming data.\n"
    "Please provide a brief advice on how I should complete this programming task. Provide 2-3 concise sentences summarizing the key coding strategy.\n"
)
BRANCH_ADVICES_PROMPT_TEMPLATE = (
    "Please help me with the following Python programming task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "{tools_desc}"
    "Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, please return `None` to abstain instead of returning an incorrect guess.\n"
    "The generated function should be robust, instead of only passing the provided examples. You are allowed use global variables to keep track of new incoming data.\n"
    "Please provide 3 advices using three different approaches. Each advice should contain 2-3 concise sentences summarizing the key coding strategy.\n"
    "Your response should be in the following format:\n"
    "Advice #1: ...\n"
    "Advice #2: ...\n"
    "Advice #3: ...\n"
)
CODEGEN_PROMPT_TEMPALTE = (
    "Please write a Python function `{api_def}` that completes the following goal:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "{tools_desc}"
    "{advice}"
    "Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, please return `None` to abstain instead of returning an incorrect guess.\n"
    "The generated function should be robust, instead of only passing the provided examples. You are allowed use global variables to keep track of new incoming data.\n"
    "Please respond with the Python implementation of `{api_def}` only. Please do not output any other responses or any explanations.\n"
    "Your response should be in the following format (the markdown format string should be included):\n"
    "```python\n"
    "def {api_def}:\n"
    "    '''Your Implementation Here.'''\n"
    "```\n"
)
EXAMPLE_PROMPT_TEMPLATE = (
    "Example #{idx}:\n"
    "Inputs:\n"
    "{inputs_desc}"
    "Output:\n"
    "{output_desc}"
    "{explanation_desc}"
)
LOGICAL_EVALUATION_PROMPT_TEMPLATE = (
    "Consider the following task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Consider the following Python function `{api_def}` that is expected to complete the above task:\n"
    "```python\n"
    "{code}"
    "```\n"
    "Please determine whether the function `{api_def}` is correct.\n"
    "Please output your judgement in the following format: first output \"Thought:\" and then thoughts on whether this function is correct or not, then output \"Answer:\" followed by a single answer \"yes\" or \"no\".\n"
)
FIXING_OR_NOT_PROMPT_TEMPLATE = (
    "Consider the following task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Consider the following Python function `{api_def}` that is expected to complete the above task:\n"
    "```python\n"
    "{code}"
    "```\n"
    "The function seems to be incorrect:\n"
    "{evaluation}"
    "Please determine whether: A. the function is actually incorrect and can be fixed. B. the function is actually incorrect and can not be easily fixed. C. the function is correct while the case evaluation is incorrect.\n"
    "Please output your judgement in the following format: first output \"Thought:\" and then thoughts on whether this function is correct or not, then output \"Answer:\" followed by a single answer \"A\", \"B\" or \"C\".\n"
)
CODE_FIXING_ADVICE_PROMPT_TEMPLATE = (
    "Consider the following task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Consider the following Python function `{api_def}` that is expected to complete the above task:\n"
    "```python\n"
    "{code}"
    "```\n"
    "The function seems to be incorrect:\n"
    "{evaluation}"
    "Please provide a brief advice on how I should fix the function. Provide 2-3 concise sentences summarizing the key coding strategy.\n"
    "The fix should be robust and general, instead of only passing the provided error cases.\n"
)
CODE_FIXING_PROMPT_TEMPLATE = (
    "Consider the following task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Consider the following Python function `{api_def}` that is expected to complete the above task:\n"
    "```python\n"
    "{code}"
    "```\n"
    "The function seems to be incorrect:\n"
    "{evaluation}"
    "{advice}"
    "Please fix the code. The fix should be robust and general, instead of only passing the provided error cases.\n"
    "Please respond with the fixed Python implementation of `{api_def}` only. Please do not output any other responses or any explanations.\n"
    "Your response should be in the following format (the markdown format string should be included):\n"
    "```python\n"
    "def {api_def}:\n"
    "    '''Your Implementation Here.'''\n"
    "```\n"
)
EXAMPLEGEN_PROMPT_TEMPLATE = (
    "Consider the following task:\n"
    "{task_desc}"
    "The input contains the following attributes:\n"
    "{inputs_desc}"
    "The function is expected to output:\n"
    "{output_desc}"
    "Examples:\n"
    "{examples_desc}"
    "Consider the following Python function `{api_def}` that is expected to complete the above task:\n"
    "```python\n"
    "{code}"
    "```\n"
    "Please provide a 3 ~ 5 test cases that you think are effective and comprehensive for determining whether the program is correct.\n"
    "You should focus on the logical correctness of the program.\n"
    "Do not design corner cases such as small floating point errors, values beyond natural ranges, behaviors around boundary, etc.\n"
    "The provided test cases should be multiple lines of compilable code, each line should be in the same format as below:\n"
    "```python\n"
    "assert {api_def} == ?\n"
    "```\n"
    "Example test cases:\n"
    "```python\n"
    "{test_desc}"
    "```\n"
    "Now, please design some new test cases. Please respond with the test cases only.\n"
)

def format_obj(obj):
    return obj if isinstance(obj, str) else repr(obj)

def format_inputs(inputs):
    return ("\n".join([f"- {k}: {v}" for k, v in inputs.items()]))+"\n"

def format_output(output):
    return f"- {output}" + "\n"

def format_example(idx, example):
    return EXAMPLE_PROMPT_TEMPLATE.format(idx=idx,
        inputs_desc=format_inputs({k:format_obj(v) for k,v in example["inputs"].items()}),
        output_desc=format_output(format_obj(example["output"])),
        explanation_desc=f"Explanation: {example['info']}" if example["info"] else "",
    )

def format_assertion(cell, example):
    inputs_repr = {k:repr(v) for k,v in example['inputs'].items()}
    output_repr = repr(example['output'])
    return f"assert {cell.api_call(**inputs_repr)} == {output_repr}"

def format_examples(examples):
    return "".join([format_example(idx, example) for idx, example in enumerate(examples)]) + ("\n" if examples else "")

def format_assertions(cell, examples):
    return "\n".join([format_assertion(cell, example) for example in examples]) + ("\n" if examples else "")

def format_tools(tools):
    return f"The following tools could be considered:\n{format_inputs(tools)}" if tools else ""

def format_advice_prompt(cell):
    return ADVICE_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        tools_desc = format_tools(cell.code_tools),
    )

def format_advices_prompt(cell):
    return BRANCH_ADVICES_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        tools_desc = format_tools(cell.code_tools),
    )

def format_advice(advice):
    return f"Hint: {advice}\n" if advice!="" else ""

def format_codegen_prompt(cell, advice=""):
    return CODEGEN_PROMPT_TEMPALTE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        tools_desc = format_tools(cell.code_tools),
        advice = format_advice(advice),
        api_def = cell.api_def(),
    )

def code_generation_single(cell, use_advice=True):
    llm = LLMCore()
    if use_advice:
        advice_prompt = format_advice_prompt(cell)
        advice = llm.Query([{"role":"user", "content":advice_prompt}])['content']
        codegen_prompt = format_codegen_prompt(cell, advice=advice)
    else:
        codegen_prompt = format_codegen_prompt(cell)
    messages = [{"role":"user", "content":codegen_prompt}]
    response = llm.Query(messages)['content']
    code = parse_code_response(response=response)
    code_doc = parse_code_doc_messages(messages=messages)
    messages = messages + [{"role":"assistant", "content":response}]
    return Cell(
        name = cell.name, code = code, doc = code_doc, messages = messages, desc = cell.desc, inputs = cell.inputs, output = cell.output, examples = cell.examples, version = 0,
    )

def code_generation_batch(cell):
    llm = LLMCore()
    advices_prompt = format_advices_prompt(cell)
    advices = llm.Query([{"role":"user", "content":advices_prompt}])['content']
    assert ("Advice #1: " in advices), (advices)
    assert ("Advice #2: " in advices), (advices)
    assert ("Advice #3: " in advices), (advices)
    advices = [
        advices.split("Advice #2: ")[0].split("Advice #1: ")[-1].strip(),
        advices.split("Advice #3: ")[0].split("Advice #2: ")[-1].strip(),
        advices.split("Advice #3: ")[-1].strip(),
    ]
    code_cells = [code_generation_single(cell, use_advice=False)]
    for idx, advice in enumerate(advices):
        codegen_prompt = format_codegen_prompt(cell, advice=advice)
        messages = [{"role":"user", "content":codegen_prompt}]
        response = llm.Query(messages)['content']
        code = parse_code_response(response=response)
        code_doc = parse_code_doc_messages(messages=messages)
        messages = messages + [{"role":"assistant", "content":response}]
        code_cells.append(Cell(
            name = cell.name, code = code, doc = code_doc, messages = messages, desc = cell.desc, inputs = cell.inputs, output = cell.output, examples = cell.examples, version = idx+1,
        ))
    return code_cells

def format_ensemble_code(cell, code_cells):
    return CODE_ENSEMBLE_CODE.format(
        code_imports = "\n".join([
            f"from . import {cell.name}_code_V{code_cell.version}\n"
            f"FUNCTIONS.append({cell.name}_code_V{code_cell.version}.{cell.name})\n"
            for code_cell in code_cells
        ]),
        api_def = cell.api_def(),
        api_copyargs = cell.api_copyargs(),
    )

def format_logical_evaluation_prompt(cell):
    return LOGICAL_EVALUATION_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        code = cell.code,
        api_def = cell.api_def(),
    )

def format_example_generation_prompt(cell):
    return EXAMPLEGEN_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        code = cell.code,
        api_def = cell.api_def(),
        test_desc = format_assertions(cell, cell.examples[:10]),
    )

def code_logical_evaluation(cell):
    prompt = format_logical_evaluation_prompt(cell)
    llm = LLMCore(); response = llm.Query([{"role":"user", "content":prompt}])['content']
    logical_evaluation = response.split('Answer:')[-1].strip('.').lower().strip()
    logical_evaluation_evidence = response.split('Thought:')[-1].split('Answer:')[0].lower().strip()
    assert(logical_evaluation in ['yes', 'no'])
    return logical_evaluation=='yes', logical_evaluation_evidence+"\n"

def test_cell(cell, example):
    env = Environment()
    env.add(ScriptCell("from seed import *"))
    env.add(cell)
    inputs_repr = {k:repr(v) for k,v in example['inputs'].items()}
    output_repr = repr(example['output'])
    env.add(ScriptCell(
        f"gt = {output_repr}\n"
        f"try:\n"
        f"    pd = {cell.api_call(**inputs_repr)}\n"
        f"    err = None\n"
        f"except Exception as e:\n"
        f"    pd = None\n"
        f"    err = str(e)\n"
        f"(err, (pd, gt))\n"
    ))
    response = env.reset()
    if not response['status']:
        correct, evidence = False, response['msg']
        raise NotImplementedError
    else:
        err, (pd, gt) = response['response']
        if (pd is not None) and (pd != gt):
            correct, evidence = False, f"Input:\n{cell.api_call(**inputs_repr)}\nExpected:{gt}\nOutput:{pd}\n"
        else:
            correct, evidence = True, None
    return correct, evidence

def code_example_evaluation(cell):
    failures = []
    for example in cell.examples:
        status, evidence = test_cell(cell, example)
        if not status:
            failures.append(evidence)
    return len(failures)==0, failures

def code_example_accuracy(cell):
    return len(code_example_evaluation(cell)[-1])/len(cell.examples)

def format_code_fixing_advice_prompt(cell, evaluation):
    return CODE_FIXING_ADVICE_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        code = cell.code,
        evaluation = evaluation,
        api_def = cell.api_def(),
    )

def format_code_fixing_prompt(cell, evaluation, advice=""):
    return CODE_FIXING_PROMPT_TEMPLATE.format(
        task_desc = cell.desc + "\n",
        inputs_desc = format_inputs(cell.inputs),
        output_desc = format_output(cell.output),
        examples_desc = format_examples(cell.examples[:10]),
        code = cell.code,
        evaluation = evaluation,
        advice = ("Hint:"+advice+"\n") if advice else "",
        api_def = cell.api_def(),
    )

def code_fixing_advice(cell, evaluation):
    advice_prompt = format_code_fixing_advice_prompt(cell, evaluation)
    llm = LLMCore(); return llm.Query([{"role":"user", "content":advice_prompt}])['content']

def code_fixing(cell, evaluation, advice="", version=-1):
    prompt = format_code_fixing_prompt(cell, evaluation, advice=advice)
    messages = [{"role":"user", "content":prompt}] # cell.messages + [{"role":"user", "content":prompt}]
    llm = LLMCore(); response = llm.Query(messages)['content']
    code = parse_code_response(response=response)
    code_doc = parse_code_doc_messages(messages=messages)
    messages = messages + [{"role":"assistant", "content":response}]
    return Cell(
        name = cell.name, code = code, doc = code_doc, messages = messages, desc = cell.desc, inputs = cell.inputs, output = cell.output, examples = cell.examples, version = version,
    )

def code_generation(cell, args):
    # Code Generation
    if args.use_ensemble:
        code_cells = code_generation_batch(cell)
    else:
        code_cells = [code_generation_single(cell)]
    m = len(code_cells)
    # Evolution
    for t in range(args.timeout):
        # 1. Branching by Fixing
        branches = [cell for cell in code_cells]
        for cell in code_cells:
            # 1.a. Logical Correctness Evaluation
            correct, evidence = code_logical_evaluation(cell)
            if not correct:
                advice = code_fixing_advice(cell, evidence)
                fixed = code_fixing(cell, evidence, advice=advice, version=cell.version+10)
                branches.append(fixed)
            # 1.b. Example-based Verification
            correct, evidences = code_example_evaluation(cell)
            evidence = "\n".join(evidences)
            if not correct:
                advice = code_fixing_advice(cell, evidence)
                fixed = code_fixing(cell, evidence, advice=advice, version=cell.version+100)
                branches.append(fixed)
        # 2. Filtering
        evaled_cells = [(code_example_accuracy(cell), cell) for cell in branches]
        code_cells = [cell for acc, cell in sorted(evaled_cells,key=lambda x:x[0],reverse=True)[:2*m]]
    return code_cells[:m]