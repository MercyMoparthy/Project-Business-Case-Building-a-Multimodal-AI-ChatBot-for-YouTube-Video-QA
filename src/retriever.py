from .embedder import model

def embed_query(query):
    return model.encode([query])[0]