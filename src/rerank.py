from sentence_transformers import SentenceTransformer, util
import torch

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def rerank(query, bm25_results, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    chunks = [r["chunk"]["chunk_text"] for r in bm25_results]
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    ranked = sorted(
        zip(bm25_results, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {"chunk": r["chunk"], "bm25_score": r["score"], "semantic_score": s}
        for r, s in ranked[:top_k]
    ]
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.retrieve import retrieve
    query = "how do prompt injection attacks work"
    bm25_results = retrieve(query, top_k=10)
    reranked = rerank(query, bm25_results, top_k=5)
    for r in reranked:
        print(f"Semantic: {r['semantic_score']:.3f} | {r['chunk']['title'][:60]}")