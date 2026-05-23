import json
from pathlib import Path

DATA_DIR = Path("data")

def chunk_paper(paper, chunk_size=200, overlap=50):
    words = paper["summary"].split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "chunk_text": " ".join(chunk_words),
            "chunk_index": len(chunks),
        })
        start += chunk_size - overlap
    return chunks

def main():
    papers_path = DATA_DIR / "raw/papers.json"
    with open(papers_path) as f:
        papers = json.load(f)

    all_chunks = []
    for paper in papers:
        chunks = chunk_paper(paper)
        all_chunks.extend(chunks)

    out_path = DATA_DIR / "chunks/chunks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Created {len(all_chunks)} chunks from {len(papers)} papers")

if __name__ == "__main__":
    main()