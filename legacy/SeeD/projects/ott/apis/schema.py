import pandas as pd
from pyheaven import *

PATH = "./data/tables/"
def GET_SCHEMA(table_id):
    table_path = pjoin(PATH, table_id+".csv")
    if not ExistFile(table_path):
        files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
        for file in files:
            if file.replace('-','_').lower().split('.')[0] == table_id.replace('-','_').lower().strip():
                table_path = pjoin(PATH, file)
                break
        if not ExistFile(table_path):
            return "The table does not exist.", True
    table = pd.read_csv(table_path)
    columns = list(table.columns)
    if columns[0] == 'Unnamed: 0':
        columns = columns[1:]
    return "The table has columns: " + ", ".join(['`'+c+'`' for c in columns]), True