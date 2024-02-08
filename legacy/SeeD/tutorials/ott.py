from seed import *

if __name__=="__main__":
    name = "data_discovery"
    project_file = "OTT-QA.jsonl"
    project = "ott"
    project_path = pjoin(get_config('projects_path'), project)
    CreateProject(project, clear=True)
    args = get_seed_args()
    
    env = Environment()
    env.add(ScriptCell(code="from seed import *"))
    env.add(ScriptCell(code=
f"""
records = LoadJson('./data/{project_file}',backend='jsonl')
ids, inputs, labels = prepare_data(records, label='tables', sample={200}, reorder="{args.reorder}", task="{name}", batch_size={args.batch_size}, seed={args.random_seed})
"""
    ))
    AddModule(project=project,
        name = "data_discovery",
        desc = "Given a natural language query, find a table that can help answer the query.",
        inputs = {
            "query": "str. The natural language query.",
        },
        output = "str. The name of the table that is found related to the given query.",
        examples = [
            {
                "inputs": {
                    "query": "How many academic staff are at the university in Budapest that has the official abbreviation BME ?",
                },
                "interaction": [
                    "Thought: I need to find tables related to the query, try with BM25 first.",
                    "Action: BM25('How many academic staff are at the university in Budapest that has the official abbreviation BME ?')",
                    "Observation: The top results: Budapest_0, Classification_society_0, Sapienza_University_of_Rome_0, List_of_University_of_Oregon_faculty_and_staff_9, List_of_University_of_Oregon_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_8, List_of_New_York_University_faculty_and_staff_1, List_of_New_York_University_faculty_and_staff_3, List_of_New_York_University_faculty_and_staff_7, List_of_University_of_Oregon_faculty_and_staff_0, Education_in_Northern_Cyprus_0, Tomography_0, Languages_with_official_status_in_India_2, National_Reporter_System_1, Case_citation_8, EUMETSAT_0, 2016_Wisconsin_Badgers_football_team_47, Budapest_Ferenc_Liszt_International_Airport_4, Turkish_population_8, Kakkonen_2"
                    "Thought: Judging by title, the 'Budapest_0' table seems to be the only table related to the query. I should confirm it by checking whether its schema is realted to the query.",
                    "Action: GET_SCHEMA('Budapest_0')",
                    "Observation: The table has columns: `Name`, `Established`, `City`, `Type`, `Students`, `Academic staff`",
                    "Thought: I thinks it is related to the query as it is about academic staff. It is now time to submit the answer.",
                    "Action: SUBMIT('Budapest_0')",
                ],
                "output": 'Budapest_0',
            },
        ],
        code_tools = {
            
        },
        query_tools = {
            "SUBMIT(table)": "Input a string. Submit the table.",
            "GET_SCHEMA(table_name)": "Input a string. Return the schema (the name of all the columns) of the given table.",
            "SEARCH_KEYWORDS(keywords)": "Input a list of keywords. Return a list containing at most 20 tables whose title or schema strictly contains at least one given keyword. The tables that contain more keywords will be ranked higher.",
            # "SEARCH_VALUE(value)": "Input a value in any type. Return a list containing at most 20 tables that contains this value (the value is fuzzy matched as a string). The tables that contain more of this value will be ranked higher.",
            "BM25(query)": "Input a string. Return a list containing at most 20 tables that are related to the query, found by running bm25 algorithm over table title and schema.",
            # "JOINT_SEARCH(keywords, value)": "Input a list of keywords and a value. Return a list containing at most 20 tables that contains both the value and at least one of the keywords.",
        },
        args = args,
    )
    
    env.add(ScriptCell(code=f"clr_llm_count()"))
    
    env.add(ScriptCell(code=
"""
from modules import *
service = {name}_integrated_class(Data(
    {args}
))
for data_input in TQDM(inputs):
    service(**data_input)
service.flush()
responses = [(list(x) if x else []) for x in service.responses]
SaveJson([d|{{'response':r}} for d,r in zip(records,responses)], "{output_file}.jsonl", backend='jsonl')
Delete("{output_file}.txt",rm=True); CreateFile("{output_file}.txt")
evaluate_list("{output_file}.txt", responses, labels)
evaluate_llm_count("{output_file}.txt")
""".format(args=repr(args),output_file=f"./{args.identifier}",name=name)
    ))
    
    env.export(pjoin(project_path, "main.py"))