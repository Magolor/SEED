
from seed import *

class data_discovery_batch_class(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.buffer = list()
        self.responses = list()
        
    def __call__(self, query):
        self.buffer.append(Data(
            query=query
        ))
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False
    
    def flush(self):
        prompt = 'Given a natural language query, find a table that can help answer the query.\nThe input contains the following attributes:\n- query: str. The natural language query.\nYou are expected to output:\n- str. The name of the table that is found related to the given query.\nExamples:\nExample #0:\nInputs:\n- query: How many academic staff are at the university in Budapest that has the official abbreviation BME ?\nOutput:\n- Budapest_0\n\nNow consider the following instances:\n{instances_desc}Please respond with the answer only, one line for each instance. Please do not output any other responses or any explanations.\nEach response should start with "Output #<index>: ". For example:\nOutput #1: ...\nOutput #2: ...\n...\n'.format(instances_desc=format_instances(self.buffer))
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
