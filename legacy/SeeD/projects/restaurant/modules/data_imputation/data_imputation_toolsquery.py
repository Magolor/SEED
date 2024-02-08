
from seed import *

class data_imputation_tools_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, name, addr, phone, type):
        '''
        Given a restaurant's information, deduce the city it is located in.
        Please do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:
        
        Interaction Examples:
        Example #0:
        Inputs:
        - name: le chardonnay (los angeles)
        - addr: 8284 melrose ave.
        - phone: 213-655-8880
        - type: french bistro
        Interactions:
        Explanation: In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:
        Inputs:
        - name: matsuhisa
        - addr: 129 n. la cienega blvd.
        - phone: 310/659-9639
        - type: asian
        Interactions:
        Explanation: Phone number has area code 310 which represents Beverly Hills, California.
        Now consider the following instance:
        {instance_desc}
        Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.
        '''
        prompt = "Given a restaurant's information, deduce the city it is located in.\nPlease do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:\n\nInteraction Examples:\nExample #0:\nInputs:\n- name: le chardonnay (los angeles)\n- addr: 8284 melrose ave.\n- phone: 213-655-8880\n- type: french bistro\nInteractions:\nExplanation: In this case, the city is contained in the restaurant's name. Also, phone number has area code 213 which represents Los Angeles, California.Example #1:\nInputs:\n- name: matsuhisa\n- addr: 129 n. la cienega blvd.\n- phone: 310/659-9639\n- type: asian\nInteractions:\nExplanation: Phone number has area code 310 which represents Beverly Hills, California.\nNow consider the following instance:\n{instance_desc}Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.\n".format(instance_desc=format_instance(
            name=name,
addr=addr,
phone=phone,
type=type
        ))
        try:
            messages = [{'role':'user', 'content':prompt}]; status = True
            while status:
                llm_response = self.llm.Query(messages)['content']
                messages.append({"role":"assistant", "content":llm_response})
                assert (llm_response.startswith("Thought:")), (llm_response)
                assert ("Action:" in llm_response), (llm_response)
                thought = llm_response.split("Thought:")[-1].split("Action:")[0].strip()
                action = llm_response.split("Action:")[-1].strip()
                print(thought)
                print(action)
                env = Environment()
                code = ScriptCell(
                    "from apis import *\n"
                    f"{action}\n"
                )
                env.add(code)
                observation, status = env.reset()['response']
                print(observation)
                if status:
                    messages.append({"role":"user", "content":f"Observation: {observation}\n"})
            print(observation)
            self.responses.append(observation)
            add_llm_count(type="llm", size=len(messages)//2, tokens=len(prompt))
        except Exception as e:
            print(e)
            self.responses.append(None)
        return True
