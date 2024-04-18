
from seed import *

class data_discovery_tools_class(object):
    def __init__(self):
        self.llm = LLMCore()
        self.clear()
    
    def clear(self):
        self.responses = list()
        
    def __call__(self, query):
        '''
        Given a natural language query, find a table that can help answer the query.
        Please do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:
        - SUBMIT(table): Input a string. Submit the table.
        - GET_SCHEMA(table_name): Input a string. Return the schema (the name of all the columns) of the given table.
        - SEARCH_KEYWORDS(keywords): Input a list of keywords. Return a list containing at most 20 tables whose title or schema strictly contains at least one given keyword. The tables that contain more keywords will be ranked higher.
        - BM25(query): Input a string. Return a list containing at most 20 tables that are related to the query, found by running bm25 algorithm over table title and schema.
        Interaction Examples:
        Example #0:
        Inputs:
        - query: How many academic staff are at the university in Budapest that has the official abbreviation BME ?
        Interactions:
        Thought: I need to find tables related to the query, try with BM25 first.
        Action: BM25('How many academic staff are at the university in Budapest that has the official abbreviation BME ?')
        Observation: The top results: Budapest_0, Classification_society_0, Sapienza_University_of_Rome_0, List_of_University_of_Oregon_faculty_and_staff_9, List_of_University_of_Oregon_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_1, List_of_New_York_University_faculty_and_staff_3, List_of_New_York_University_faculty_and_staff_7, List_of_University_of_Oregon_faculty_and_staff_0, Education_in_Northern_Cyprus_0, Tomography_0, Languages_with_official_status_in_India_2, National_Reporter_System_1, Case_citation_8, EUMETSAT_0, 2016_Wisconsin_Badgers_football_team_47, Budapest_Ferenc_Liszt_International_Airport_4, Turkish_population_8, Kakkonen_2Thought: Judging by title, the 'Budapest_0' table seems to be the only table related to the query. I should confirm it by checking whether its schema is realted to the query.
        Action: GET_SCHEMA('Budapest_0')
        Observation: The table has columns: `Name`, `Established`, `City`, `Type`, `Students`, `Academic staff`
        Thought: I thinks it is related to the query as it is about academic staff. It is now time to submit the answer.
        Action: SUBMIT('Budapest_0')
        
        Now consider the following instance:
        {instance_desc}
        Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.
        '''
        prompt = "Given a natural language query, find a table that can help answer the query.\nPlease do not directly answer the problem. This should be an interactive process. You are allowed to take one of the following actions:\n- SUBMIT(table): Input a string. Submit the table.\n- GET_SCHEMA(table_name): Input a string. Return the schema (the name of all the columns) of the given table.\n- SEARCH_KEYWORDS(keywords): Input a list of keywords. Return a list containing at most 20 tables whose title or schema strictly contains at least one given keyword. The tables that contain more keywords will be ranked higher.\n- BM25(query): Input a string. Return a list containing at most 20 tables that are related to the query, found by running bm25 algorithm over table title and schema.\nInteraction Examples:\nExample #0:\nInputs:\n- query: How many academic staff are at the university in Budapest that has the official abbreviation BME ?\nInteractions:\nThought: I need to find tables related to the query, try with BM25 first.\nAction: BM25('How many academic staff are at the university in Budapest that has the official abbreviation BME ?')\nObservation: The top results: Budapest_0, Classification_society_0, Sapienza_University_of_Rome_0, List_of_University_of_Oregon_faculty_and_staff_9, List_of_University_of_Oregon_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_1, List_of_New_York_University_faculty_and_staff_3, List_of_New_York_University_faculty_and_staff_7, List_of_University_of_Oregon_faculty_and_staff_0, Education_in_Northern_Cyprus_0, Tomography_0, Languages_with_official_status_in_India_2, National_Reporter_System_1, Case_citation_8, EUMETSAT_0, 2016_Wisconsin_Badgers_football_team_47, Budapest_Ferenc_Liszt_International_Airport_4, Turkish_population_8, Kakkonen_2Thought: Judging by title, the 'Budapest_0' table seems to be the only table related to the query. I should confirm it by checking whether its schema is realted to the query.\nAction: GET_SCHEMA('Budapest_0')\nObservation: The table has columns: `Name`, `Established`, `City`, `Type`, `Students`, `Academic staff`\nThought: I thinks it is related to the query as it is about academic staff. It is now time to submit the answer.\nAction: SUBMIT('Budapest_0')\n\nNow consider the following instance:\n{instance_desc}Your respond should strictly follow this format: first output `Thought:` followed by your thought process, then output `Action:` followed by one of the actions mentioned above.\n".format(instance_desc=format_instance(
            query=query
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
