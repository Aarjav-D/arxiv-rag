import requests
import json
import time
from pathlib import Path

BASE_URL = "http://export.arxiv.org/api/query"
DATA_DIR = Path("data/raw")

QUERIES = [
    "prompt injection attacks large language models",
    "retrieval augmented generation",
    "information retrieval neural",
    "AI safety alignment",
]

def fetch_papers(query, max_results=25):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.text

def parse_papers(xml_text):
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        paper = {
            "id": entry.find("atom:id", ns).text.strip(),
            "title": entry.find("atom:title", ns).text.strip(),
            "summary": entry.find("atom:summary", ns).text.strip(),
            "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)],
        }
        papers.append(paper)
    return papers

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_papers = []
    for query in QUERIES:
        print(f"Fetching: {query}")
        xml = fetch_papers(query)
        papers = parse_papers(xml)
        all_papers.extend(papers)
        time.sleep(1)
    seen = set()
    unique = []
    for p in all_papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    out_path = DATA_DIR / "papers.json"
    with open(out_path, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Saved {len(unique)} unique papers to {out_path}")
if __name__ == "__main__":
    main()