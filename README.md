# Unhackathon RAG Starter

A minimal retrieval-augmented generation (RAG) demo that builds a FAISS index over the `.txt` and `.pdf` files in `docs/`, retrieves the most relevant chunks, and asks a hosted LLM on Hugging Face Inference API to answer your question strictly from that context.

## Features
- CPU-only workflow with `sentence-transformers` embeddings (`intfloat/e5-small-v2`).
- Chunking (~180 words with overlap) and FAISS inner-product search.
- Streamlit UI that cites sources and shows retrieved passages.
- Hosted generation via Hugging Face Inference API (`Qwen/Qwen2.5-1.5B-Instruct` by default).
- Ready for local use (macOS/Windows) and deployment to Hugging Face Spaces or Render.

---

## 1. Local Setup

### Prerequisites
- Python 3.10+
- Hugging Face Inference API token with access to the chosen model.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare Your Documents
1. Place `.txt` and `.pdf` files inside `docs/`.
2. For a quick smoke test, create `docs/topic.txt` with a few sentences.

### Build the FAISS Index
```bash
python build_index.py
```
This writes `index/faiss.index` and `index/chunks.json`.

### Configure Environment Variables
Set the Hugging Face API token (and optional custom model).

macOS / Linux:
```bash
export HF_TOKEN="hf_your_token_here"
export HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"  # optional override
```

Windows (PowerShell):
```powershell
$env:HF_TOKEN = "hf_your_token_here"
$env:HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # optional override
```

### Run the Streamlit App
```bash
streamlit run app.py
```
Open the local URL displayed in the terminal, ask questions, and verify citations.

---

## 2. Deploy to Hugging Face Spaces
1. Create a new **Space** (Streamlit template) on Hugging Face.
2. Upload the repo files (or connect the GitHub repository).
3. In the Space settings, add a **Secret** named `HF_TOKEN` (and optionally `HF_MODEL`).
4. Redeploy; the Streamlit app will launch automatically in the Space.

---

## 3. Deploy to Render (Optional)
1. Create a new **Web Service** and connect to your repository.
2. Select a Python 3.10+ environment.
3. Set the **Build Command** to:
   ```
   pip install -r requirements.txt
   ```
4. Set the **Start Command** to:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add environment variables `HF_TOKEN` (required) and optionally `HF_MODEL`.

---

## Troubleshooting
- **"No .txt or .pdf files"**: Add at least one text file to `docs/` before running `build_index.py`.
- **"No chunks produced"**: PDF pages without extractable text may be scanned images; try providing a plain text file.
- **Weak or missing citations**: Increase the *Top-K passages* slider in the app or rebuild the index with a larger chunk size.
- **HF errors / 401**: Confirm that `HF_TOKEN` has access to the requested model.
- **Timeouts**: Use a smaller model via `HF_MODEL` or retry later; hosted inference endpoints can be busy.

Enjoy building on this RAG starter!
