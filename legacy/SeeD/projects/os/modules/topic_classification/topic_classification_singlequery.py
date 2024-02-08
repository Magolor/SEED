
from seed import *

class topic_classification_single_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, text):
        '''
        Given a tweet message, determine whether it is hatespeech or benign.
        The input contains the following attributes:
        - text: str. The tweet message.
        You are expected to output:
        - int. Output 0 for hatespeech, 1 for benign.
        Examples:
        Example #0:
        Inputs:
        - text: !!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out...
        Output:
        - 1
        Example #1:
        Inputs:
        - text: " &amp; you might not get ya bitch back &amp; thats that "
        Output:
        - 0
        Example #2:
        Inputs:
        - text: "@CB_Baby24: @white_thunduh alsarabsss" hes a beaner smh you can tell hes a mexican
        Output:
        - 0
        
        Now consider the following instance:
        {instance_desc}
        Please respond with the answer only. Please do not output any other responses or any explanations.
        '''
        prompt = 'Given a tweet message, determine whether it is hatespeech or benign.\nThe input contains the following attributes:\n- text: str. The tweet message.\nYou are expected to output:\n- int. Output 0 for hatespeech, 1 for benign.\nExamples:\nExample #0:\nInputs:\n- text: !!! RT @mayasolovely: As a woman you shouldn\'t complain about cleaning up your house. &amp; as a man you should always take the trash out...\nOutput:\n- 1\nExample #1:\nInputs:\n- text: " &amp; you might not get ya bitch back &amp; thats that "\nOutput:\n- 0\nExample #2:\nInputs:\n- text: "@CB_Baby24: @white_thunduh alsarabsss" hes a beaner smh you can tell hes a mexican\nOutput:\n- 0\n\nNow consider the following instance:\n{instance_desc}Please respond with the answer only. Please do not output any other responses or any explanations.\n'.format(instance_desc=format_instance(
            text=text
        ))
        try:
            response = self.llm.Query([{'role':'user', 'content':prompt}])['content']
            self.responses.append(response)
        except:
            self.responses.append(None)
        add_llm_count(type="llm", size=1, tokens=len(prompt))
        return True
