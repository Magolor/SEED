import pandas as pd
from pyheaven import *
from Levenshtein import distance as LevenshteinDistance

PATH = "./data/tables/"
def SEARCH_VALUE(value):
    value = value.lower().strip()
    files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
    results = []
    for file in files:
        table = pd.read_csv(pjoin(PATH, file))
        values = [str(x).lower().strip() for x in table.values.flatten()]
        c = 0
        for v in values:
            if LevenshteinDistance(v, value)/len(value) <= 0.2:
                c += 1
        results.append((c, file.split('.csv')[0]))
    results = [r[1] for r in sorted(results, reverse=True)[:20] if r[0]>0]
    if len(results) == 0:
        return "No results found.", True
    return "The top results: "+ ", ".join(results), True