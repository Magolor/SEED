
from seed import *

class data_discovery_single_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, query):
        '''
        Given a natural language query, find a table that can help answer the query.
        The input contains the following attributes:
        - query: str. The natural language query.
        You are expected to output:
        - str. The name of the table that is found related to the given query.
        Examples:
        Example #0:
        Inputs:
        - query: How many academic staff are at the university in Budapest that has the official abbreviation BME ?
        Output:
        - Budapest_0
        
        Now consider the following instance:
        {instance_desc}
        Please respond with the answer only. Please do not output any other responses or any explanations.
        '''
        prompt = 'Given a natural language query, find a table that can help answer the query.\nThe input contains the following attributes:\n- query: str. The natural language query.\nYou are expected to output:\n- str. The name of the table that is found related to the given query.\nExamples:\nExample #0:\nInputs:\n- query: How many academic staff are at the university in Budapest that has the official abbreviation BME ?\nOutput:\n- Budapest_0\n\nNow consider the following instance:\n{instance_desc}Please respond with the answer only. Please do not output any other responses or any explanations.\n'.format(instance_desc=format_instance(
            query=query
        ))
        try:
            response = self.llm.Query([{'role':'user', 'content':prompt}])['content']
            self.responses.append(response)
        except Exception as e:
            print(e)
            self.responses.append(None)
        add_llm_count(type="llm", size=1, tokens=len(prompt))
        return True
