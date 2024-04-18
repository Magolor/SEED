import pandas as pd
from pyheaven import *
from Levenshtein import distance as LevenshteinDistance

PATH = "./data/tables/"
def JOINT_SEARCH(keywords, value):
    value = value.lower().strip()
    files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
    results1 = []
    for file in files:
        table = pd.read_csv(pjoin(PATH, file))
        values = [str(x).lower().strip() for x in table.values.flatten()]
        c = 0
        for v in values:
            if LevenshteinDistance(v, value)/len(value) <= 0.2:
                c += 1
        results1.append((c, file.split('.csv')[0]))
    results1 = [r[1] for r in sorted(results1, reverse=True) if r[0]>0]
    
    keywords = [k.lower().strip() for k in keywords]
    files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
    results2 = []
    for file in files:
        table = pd.read_csv(pjoin(PATH, file))
        title = file.split('.csv')[0]
        columns = list(table.columns)
        schema = ' '.join(re.split(r'\s+|-|_', title)) + " " + (" ".join(columns))
        tokens = schema.lower().split()
        c = 0
        for key in keywords:
            for t in tokens:
                if key in t:
                    c += 1
                    break
        results2.append((c, title))
    results2 = [r[1] for r in sorted(results2, reverse=True) if r[0]>0]
    
    joint_results = [r for r in results1 if r in results2]
    if len(joint_results) == 0:
        return "No results found.", True
    return "The top results: "+ ", ".join(joint_results), True