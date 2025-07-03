from .retriever import embed_query
from .vector_store import search_index

def get_answer(user_query, index, chunks):
    query_vec = embed_query(user_query)
    indices = search_index(index, query_vec)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(relevant_chunks)    
    return "This is a placeholder answer using retrieved context."