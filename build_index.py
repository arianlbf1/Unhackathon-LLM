import os, json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
from tqdm import tqdm

DOCS_DIR = "docs"
OUT_DIR = "index"
EMBED_MODEL = "intfloat/e5-small-v2"

CHUNK_WORDS = 180
OVERLAP_WORDS = 40

def read_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "meta": {"source": os.path.basename(path), "page": None}}]

def read_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        out.append({"text": t, "meta": {"source": os.path.basename(path), "page": i+1}})
    return out

def chunk_words(text: str, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i+size]).strip()
        if piece:
            chunks.append(piece)
        i += max(1, size - overlap)
    return chunks

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    if not os.path.isdir(DOCS_DIR):
        print(f"Missing folder: {DOCS_DIR}")
        return

    fnames = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith((".txt",".pdf"))]
    if not fnames:
        print(f"No .txt or .pdf files in {DOCS_DIR}. Add a small .txt to test.")
        return

    print(f"Scanning {DOCS_DIR} ...")
    for fname in fnames:
        path = os.path.join(DOCS_DIR, fname)
        docs = read_txt(path) if fname.lower().endswith(".txt") else read_pdf(path)
        for d in docs:
            for c in chunk_words(d["text"]):
                all_chunks.append({
                    "text": c,
                    "source": d["meta"]["source"],
                    "page": d["meta"]["page"]
                })

    if not all_chunks:
        print("No chunks produced (PDFs might be scanned images). Try a .txt first.")
        return

    print(f"Embedding {len(all_chunks)} chunks ...")
    texts = [c["text"] for c in all_chunks]
    embs = model.encode([f"passage: {t}" for t in texts],
                        normalize_embeddings=True, batch_size=64)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))
    with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("Index built ✅  Files saved to index/")

if __name__ == "__main__":
    main()
