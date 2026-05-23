import pickle
from pathlib import Path

DATA_DIR = Path("data")

def load_index():
    with open(DATA_DIR / "index/bm25.pkl", "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]

def retrieve(query, top_k=10):
    bm25, chunks = load_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": scores[idx],
        })
    return results

if __name__ == "__main__":
    query = "how do prompt injection attacks work"
    results = retrieve(query, top_k=5)
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['chunk']['title'][:60]}")