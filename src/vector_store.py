import faiss
import numpy as np

def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_vector, top_k=5):
    distances, indices = index.search(query_vector, top_k)
    return indices