from rank_bm25 import BM25Okapi
import pandas as pd
from pyheaven import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('stopwords')
def preprocess_text(text):
    # Tokenize the text
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    tokens = word_tokenize(text)

    # Lowercase the tokens
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def flatten_list(nested_list):
    return [str(item) for sublist in nested_list for item in sublist]

def search_tables(query, tables):
    titles = [table['title'] for table in tables]
    attributes = [' '.join(table['attributes']) for table in tables]
    values = [' '.join(flatten_list(table['values'])) for table in tables]
    corpus = [' '.join(preprocess_text(t) + preprocess_text(a)) for t,a,v in zip(titles,attributes,values)]
    tokenized_corpus = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = (' '.join(preprocess_text(query))).split()
    doc_scores = bm25.get_scores(tokenized_query)
    sorted_indices = sorted(range(len(doc_scores)), key=lambda k: doc_scores[k], reverse=True)
    related_tables = [tables[idx]['title'] for idx in sorted_indices][:20]
    return related_tables

PATH = "./data/tables/"
def BM25(query):
    files = [f for f in ListFiles(PATH) if f.endswith(".csv")]
    tables = [pd.read_csv(pjoin(PATH, f)) for f in files]
    tables = [{'title': f.split('.csv')[0], 'attributes': list(table.columns), 'values': table.values.tolist()} for f, table in zip(files, tables)]
    related_tables = search_tables(query, tables)
    return "The top results: "+ ", ".join(related_tables), True