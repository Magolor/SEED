from .utils import *

@LinguaManga.register
class NoneValidator(Data):
    __type__: str = 'validator-none'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.__type__ = self.__type__
    def validate(self, init_cell, module):
        return init_cell

def verify_cell(cell, module, prev_cells=list(), extra_examples=list()):
    if len(module.examples + extra_examples) == 0:
        return []
    env = Environment(env_cells = prev_cells + [cell])
    failures = []
    for example in module.examples + extra_examples:
        inputs_repr = {k:repr(v) for k,v in example['inputs'].items()}
        output_repr = repr(example['outputs']['output'])
        test_cell = Cell(code=
            f"gt = {output_repr}\n"
            f"try:\n"
            f"    pd = {module.api_call(**inputs_repr)}\n"
            f"    try:\n"
            f"        assert (pd == gt), 'AssertionError. Output: ' + str(pd) + '.'\n"
            f"        err = None\n"
            f"    except Exception as e:\n"
            f"        err = str(e)\n"
            f"except Exception as e:\n"
            f"    pd = None\n"
            f"    err = str(e)\n"
            f"data = err, (pd, gt)\n"
        )
        assert_command = f"assert({module.api_call(**inputs_repr)} == {output_repr})"
        response = env.execute(test_cell, reset=True)
        if not response['status']:
            print(response['cell'].code)
            raise NotImplementedError
        err, (pd, gt) = env.data
        if err is not None:
            failures.append(
                {
                    'inputs': example['inputs'],
                    'outputs': example['outputs'],
                    'test_cell': test_cell,
                    'test_case': assert_command,
                    'msg': err,
                    'pd': pd,
                    'gt': gt,
                }
            )
            if len(failures) >= 5:
                return failures
    env.exit()
    return failures
    
def fuzzy_verify_cell(cell, module, prev_cells=list(), extra_examples=list()):
    if len(module.examples + extra_examples) == 0:
        return []
    env = Environment(env_cells = prev_cells + [cell])
    failures = []
    for example in module.examples + extra_examples:
        inputs_repr = {k:repr(v) for k,v in example['inputs'].items()}
        output_repr = repr(example['outputs']['output'])
        test_cell = Cell(code=
            f"gt = {output_repr}.lower()\n"
            f"try:\n"
            f"    pd = {module.api_call(**inputs_repr)}.lower()\n"
            f"    try:\n"
            f"        assert (pd == gt) or (pd in gt) or (gt in pd), 'AssertionError. Output: ' + str(pd) + '.'\n"
            f"        err = None\n"
            f"    except Exception as e:\n"
            f"        err = str(e)\n"
            f"except Exception as e:\n"
            f"    pd = None\n"
            f"    err = str(e)\n"
            f"data = err, (pd, gt)\n"
        )
        assert_command = f"pd = {module.api_call(**inputs_repr)}.lower(); gt = {output_repr}.lower(); assert((pd == gt) or (pd in gt) or (gt in pd))"
        assert_command_hint = f"assert({module.api_call(**inputs_repr)} == {output_repr})"
        response = env.execute(test_cell, reset=True)
        if not response['status']:
            print(response['cell'].code)
            raise NotImplementedError
        err, (pd, gt) = env.data
        if err is not None:
            failures.append(
                {
                    'inputs': example['inputs'],
                    'outputs': example['outputs'],
                    'test_cell': test_cell,
                    'test_case': assert_command,
                    'test_case_hint': assert_command_hint,
                    'msg': err,
                    'pd': pd,
                    'gt': gt,
                }
            )
            if len(failures) >= 5:
                return failures
    env.exit()
    return failures

@LinguaManga.register
class ExampleVerifier(NoneValidator):
    __type__: str = 'validator-verifier'
    def validate(self, init_cell, module, prev_cells=list(), extra_examples=list()):
        failures = verify_cell(init_cell, module, prev_cells=prev_cells, extra_examples=extra_examples)
        cell = init_cell
        timeout = 5
        while (len(failures) > 0) and (timeout > 0):
            print(cell.code)
            print(failures)
            messages = list(cell.messages)
            err_prompt = "Your code does not pass the following test case(s):\n" + "\n".join(
                add_indent(f"Case: {failure['test_case_hint']}\nError: {failure['msg']}\n") for failure in failures
            ) + "\n" + "Please fix your code. Instead of only passing these examples, please make sure the fixed code generalizes to all similar cases and fix other potential errors. Please respond with the fixed code only.\n"
            messages.append({"role": "user", "content": err_prompt})
            cell = LLMGC(messages=messages)
            failures = verify_cell(cell, module, prev_cells=prev_cells, extra_examples=extra_examples)
            timeout -= 1
        return cell

@LinguaManga.register
class FuzzyExampleVerifier(NoneValidator):
    __type__: str = 'validator-verifier_fuzzy'
    def validate(self, init_cell, module, prev_cells=list(), extra_examples=list()):
        failures = fuzzy_verify_cell(init_cell, module, prev_cells=prev_cells, extra_examples=extra_examples)
        cell = init_cell
        timeout = 5
        while (len(failures) > 0) and (timeout > 0):
            print(cell.code)
            print(failures)
            messages = list(cell.messages)
            err_prompt = "Your code does not pass the following test case(s):\n" + "\n".join(
                add_indent(f"Case: {failure['test_case']}\nError: {failure['msg']}\n") for failure in failures
            ) + "\n" + "Please fix your code. Instead of only passing these examples, please make sure the fixed code generalizes to all similar cases and fix other potential errors. Please respond with the fixed code only.\n"
            messages.append({"role": "user", "content": err_prompt})
            cell = LLMGC(messages=messages)
            failures = fuzzy_verify_cell(cell, module, prev_cells=prev_cells, extra_examples=extra_examples)
            timeout -= 1
        return cell

def example_gen(cell, module):
    existing_examples = []
    for example in module.examples:
        inputs_repr = {k:repr(v) for k,v in example['inputs'].items()}
        output_repr = repr(example['outputs']['output'])
        assert_command = f"assert({module.api_call(**inputs_repr)} == {output_repr})"
        existing_examples.append(assert_command)
    example_gen_prompt = (
        f"Task description: {module.task_desc}\n" +
        f"Here is a program that attempts to solve this task:\n" +
        cell.messages[-1]['content'] + "\n" +
        f"Please provide a 3 ~ 5 test cases that you think are effective and comprehensive for determining whether the program is correct.\n" +
        f"Example program:\n" +
        f"def has_close_elements(numbers, threshold):\n" +
        f"    for idx, elem in enumerate(numbers):\n" +
        f"        for idx2, elem2 in enumerate(numbers):\n" +
        f"            if idx != idx2:\n" +
        f"                distance = abs(elem - elem2)\n" +
        f"                if distance < threshold:\n" +
        f"                    return True\n" +
        f"    return False\n" +
        f"Example test cases:\n" +
        f"assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n" +
        f"assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n" +
        f"assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n" +
        f"assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n" +
        f"assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n" +
        f"assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n" +
        f"assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n" +
        f"\n" +
        f"You should focus on the logical correctness of the program.\n"
        f"Do not design corner cases such as small floating point errors, values beyond natural ranges, behaviors around boundary, etc.\n" +
        f"The provided test cases should be multiple lines of compilable code, each line should be in the same formata as below:\n" +
        f"assert({module.api_call(**{k:'?' for k in module.inputs})} == ?)\n" +
        f"Existing test cases for reference:\n" +
        "\n".join(example for example in existing_examples) + "\n" +
        f"Now, please provide your diverse test cases. Please respond with the test cases only.\n"
    )
    print(example_gen_prompt)
    example_codes = LLMQuery(messages=[{"role": "user", "content": example_gen_prompt}]).strip()
    print(example_codes)
    parsed_examples = []
    for example_code in example_codes.split('\n'):
        try:
            code = example_code.strip()
            assert(code.startswith('assert(') and code.endswith(')') and ' == ' in code)
            api_call, output = code[7:-1].strip().split(' == ')
            assert(api_call.startswith('M') and api_call.endswith(')'))
            args = api_call[api_call.index('(')+1:-1]
            example = {}
            for key, next_key in zip(module.inputs, module.inputs[1:]+[None]):
                assert(key in args)
                value_start = args.index(key) + len(key) + 1
                value_end = (args.index(next_key) - 2) if next_key is not None else 0
                value = args[value_start:value_end] if value_end > 0 else args[value_start:]
                example[key] = eval(value)
            parsed_examples.append({
                'inputs': example,
                'outputs': {'output': eval(output)},
            })
        except:
            pass
    return parsed_examples

@LinguaManga.register
class AdversarialVerifier(ExampleVerifier):
    __type__: str = 'validator-adversarial'
    def validate(self, init_cell, module):
        verified_cell = super().validate(init_cell, module)
        extra_examples = example_gen(verified_cell, module)
        verified_cell = super().validate(verified_cell, module, extra_examples=extra_examples)
        return verified_cell