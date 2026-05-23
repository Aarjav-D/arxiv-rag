import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")

def load_chunks():
    with open(DATA_DIR / "chunks/chunks.json") as f:
        return json.load(f)

def build_index(chunks):
    corpus = [chunk["chunk_text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25

def main():
    chunks = load_chunks()
    print(f"Building BM25 index over {len(chunks)} chunks...")
    bm25 = build_index(chunks)
    
    index_path = DATA_DIR / "index/bm25.pkl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    main()