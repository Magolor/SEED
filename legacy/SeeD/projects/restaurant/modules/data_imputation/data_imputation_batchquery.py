
from seed import *

class data_imputation_batch_class(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.buffer = list()
        self.responses = list()
        
    def __call__(self, name, addr, phone, type):
        self.buffer.append(Data(
            name=name,
addr=addr,
phone=phone,
type=type
        ))
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False
    
    def flush(self):
        prompt = 'Given a restaurant\'s information, deduce the city it is located in.\nThe input contains the following attributes:\n- name: str. The name of the restaurant.\n- addr: str. The address of the restaurant.\n- phone: str. The phone number of the restaurant.\n- type: str. The food type of the restaurant.\nYou are expected to output:\n- str. The city the restaurant is in.\nExamples:\nExample #0:\nInputs:\n- name: le chardonnay (los angeles)\n- addr: 8284 melrose ave.\n- phone: 213-655-8880\n- type: french bistro\nOutput:\n- los angeles\nExplanation: In this case, the city is contained in the restaurant\'s name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:\nInputs:\n- name: matsuhisa\n- addr: 129 n. la cienega blvd.\n- phone: 310/659-9639\n- type: asian\nOutput:\n- beverly hills\nExplanation: Phone number has area code 310 which represents Beverly Hills, California.\nNow consider the following instances:\n{instances_desc}Please respond with the answer only, one line for each instance. Please do not output any other responses or any explanations.\nEach response should start with "Output #<index>: ". For example:\nOutput #1: ...\nOutput #2: ...\n...\n'.format(instances_desc=format_instances(self.buffer))
        try:
            response = self.llm.Query([{'role':'user', 'content':prompt}])['content']
            lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
            assert (len(lines) == len(self.buffer)), "Incorrect Responses!"
            values = []
            for idx, line in enumerate(lines, 1):
                assert (line.startswith(f"Output #{idx}: ")), "Incorrect Responses!"
                value_repr = line.split(f"Output #{idx}: ")[-1].strip()
                try:
                    value = eval(value_repr)
                except:
                    value = value_repr
                values.append(value)
            self.responses.extend(values)
        except:
            self.responses.extend([None for _ in self.buffer])
        if len(self.buffer)>0:
            add_llm_count(type="llm", size=len(self.buffer), tokens=len(prompt))
        self.buffer = list()
