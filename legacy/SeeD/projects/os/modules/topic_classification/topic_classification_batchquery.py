
from seed import *

class topic_classification_batch_class(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.buffer = list()
        self.responses = list()
        
    def __call__(self, text):
        self.buffer.append(Data(
            text=text
        ))
        if len(self.buffer) >= self.batch_size:
            self.flush()
            return True
        return False
    
    def flush(self):
        prompt = 'Given a tweet message, determine whether it is hatespeech or benign.\nThe input contains the following attributes:\n- text: str. The tweet message.\nYou are expected to output:\n- int. Output 0 for hatespeech, 1 for benign.\nExamples:\nExample #0:\nInputs:\n- text: !!! RT @mayasolovely: As a woman you shouldn\'t complain about cleaning up your house. &amp; as a man you should always take the trash out...\nOutput:\n- 1\nExample #1:\nInputs:\n- text: " &amp; you might not get ya bitch back &amp; thats that "\nOutput:\n- 0\nExample #2:\nInputs:\n- text: "@CB_Baby24: @white_thunduh alsarabsss" hes a beaner smh you can tell hes a mexican\nOutput:\n- 0\n\nNow consider the following instances:\n{instances_desc}Please respond with the answer only, one line for each instance. Please do not output any other responses or any explanations.\nEach response should start with "Output #<index>: ". For example:\nOutput #1: ...\nOutput #2: ...\n...\n'.format(instances_desc=format_instances(self.buffer))
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
