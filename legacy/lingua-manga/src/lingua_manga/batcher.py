from .utils import *
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def get_even_clusters(embeds, C):
    n = len(embeds)
    embeds = np.array(embeds)
    kmeans = KMeans(n_clusters=C, random_state=0).fit(embeds)
    centers = kmeans.cluster_centers_.reshape(-1, 1, embeds.shape[-1]).repeat(n//C, 1).reshape(-1, embeds.shape[-1])
    distance_matrix = cdist(embeds, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//(n//C)
    ids_batched = [[j for j in range(n) if clusters[j]==i] for i in range(C)]
    return ids_batched

def find_closest(embeds, embeds_dict):
    d = 1e9; j = -1
    for k, e in embeds_dict.items():
        x = min([np.linalg.norm(embed-e) for embed in embeds])
        if x < d:
            d = x; j = k
    return j

def find_farthest(embeds, embeds_dict):
    d = -1e9; j = -1
    for k, e in embeds_dict.items():
        x = min([np.linalg.norm(embed-e) for embed in embeds])
        if x > d:
            d = x; j = k
    return j

class SimilarBatch(object):
    def __init__(self, model="sentence-transformers/all-MiniLM-L12-v2"):
        self.tokenizer, self.model = load_model(model)
    
    def order(self, keys, batching=32):
        n = len(keys); B = batching; S = n//B
        embeds = [embedding(key, self.tokenizer, self.model).detach().numpy() for key in keys]
        ids_batched = get_even_clusters(embeds, S); ids = []
        for i in range(S):
            ids.extend(ids_batched[i])
        inv_ids = [0]*n
        for i, id in enumerate(ids):
            inv_ids[id] = i
        return ids, inv_ids

class DiverseBatch(object):
    def __init__(self, model="sentence-transformers/all-MiniLM-L12-v2"):
        self.tokenizer, self.model = load_model(model)
    
    def order(self, keys, batching=32):
        n = len(keys); B = batching; S = n//B
        embeds = [embedding(key, self.tokenizer, self.model).detach().numpy() for key in keys]
        ids_batched = get_even_clusters(embeds, B); ids = []
        for i in range(S):
            for j in range(B):
                ids.append(ids_batched[j][i])
        for i in range(S):
            for j in range(B):
                ids.append(ids_batched[j][i])
        inv_ids = [0]*n
        for i, id in enumerate(ids):
            inv_ids[id] = i
        return ids, inv_ids

class ClosestBatch(object):
    def __init__(self, model="sentence-transformers/all-MiniLM-L12-v2"):
        self.tokenizer, self.model = load_model(model)
    
    def order(self, keys, batching=32):
        n = len(keys); B = batching; S = n//B
        embeds = [embedding(key, self.tokenizer, self.model).detach().numpy() for key in keys]
        embeds_dict = {i:embed for i, embed in enumerate(embeds)}; ids = []
        for i in range(S):
            idx, embed = list(embeds_dict.items())[0]; del embeds_dict[idx]
            batched_embeds = [embed]; ids.append(idx)
            while len(batched_embeds) < B:
                j = find_closest(batched_embeds, embeds_dict)
                batched_embeds.append(embeds_dict[j])
                ids.append(j)
                del embeds_dict[j]
        inv_ids = [0]*n
        for i, id in enumerate(ids):
            inv_ids[id] = i
        return ids, inv_ids

class FarthestBatch(object):
    def __init__(self, model="sentence-transformers/all-MiniLM-L12-v2"):
        self.tokenizer, self.model = load_model(model)
    
    def order(self, keys, batching=32):
        n = len(keys); B = batching; S = n//B
        embeds = [embedding(key, self.tokenizer, self.model).detach().numpy() for key in keys]
        embeds_dict = {i:embed for i, embed in enumerate(embeds)}; ids = []
        for i in range(S):
            idx, embed = list(embeds_dict.items())[0]; del embeds_dict[idx]
            batched_embeds = [embed]; ids.append(idx)
            while len(batched_embeds) < B:
                j = find_farthest(batched_embeds, embeds_dict)
                batched_embeds.append(embeds_dict[j])
                ids.append(j)
                del embeds_dict[j]
        inv_ids = [0]*n
        for i, id in enumerate(ids):
            inv_ids[id] = i
        return ids, inv_ids