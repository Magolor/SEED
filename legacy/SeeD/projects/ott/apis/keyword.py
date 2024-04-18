import pandas as pd
from pyheaven import *
import re

PATH = "./data/tables/"
def SEARCH_KEYWORDS(keywords):
    keywords = [k.lower().strip() for k in keywords]
    files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
    results = []
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
        results.append((c, title))
    if len(results) == 0:
        return "No results found.", True
    return "The top results: "+ ", ".join([r[1] for r in sorted(results, reverse=True)[:20]]), True