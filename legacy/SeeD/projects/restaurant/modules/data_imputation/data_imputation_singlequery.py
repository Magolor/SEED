
from seed import *

class data_imputation_single_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, name, addr, phone, type):
        '''
        Given a restaurant's information, deduce the city it is located in.
        The input contains the following attributes:
        - name: str. The name of the restaurant.
        - addr: str. The address of the restaurant.
        - phone: str. The phone number of the restaurant.
        - type: str. The food type of the restaurant.
        You are expected to output:
        - str. The city the restaurant is in.
        Examples:
        Example #0:
        Inputs:
        - name: le chardonnay (los angeles)
        - addr: 8284 melrose ave.
        - phone: 213-655-8880
        - type: french bistro
        Output:
        - los angeles
        Explanation: In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:
        Inputs:
        - name: matsuhisa
        - addr: 129 n. la cienega blvd.
        - phone: 310/659-9639
        - type: asian
        Output:
        - beverly hills
        Explanation: Phone number has area code 310 which represents Beverly Hills, California.
        Now consider the following instance:
        {instance_desc}
        Please respond with the answer only. Please do not output any other responses or any explanations.
        '''
        prompt = "Given a restaurant's information, deduce the city it is located in.\nThe input contains the following attributes:\n- name: str. The name of the restaurant.\n- addr: str. The address of the restaurant.\n- phone: str. The phone number of the restaurant.\n- type: str. The food type of the restaurant.\nYou are expected to output:\n- str. The city the restaurant is in.\nExamples:\nExample #0:\nInputs:\n- name: le chardonnay (los angeles)\n- addr: 8284 melrose ave.\n- phone: 213-655-8880\n- type: french bistro\nOutput:\n- los angeles\nExplanation: In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:\nInputs:\n- name: matsuhisa\n- addr: 129 n. la cienega blvd.\n- phone: 310/659-9639\n- type: asian\nOutput:\n- beverly hills\nExplanation: Phone number has area code 310 which represents Beverly Hills, California.\nNow consider the following instance:\n{instance_desc}Please respond with the answer only. Please do not output any other responses or any explanations.\n".format(instance_desc=format_instance(
            name=name,
addr=addr,
phone=phone,
type=type
        ))
        try:
            response = self.llm.Query([{'role':'user', 'content':prompt}])['content']
            self.responses.append(response)
        except Exception as e:
            print(e)
            self.responses.append(None)
        add_llm_count(type="llm", size=1, tokens=len(prompt))
        return True
