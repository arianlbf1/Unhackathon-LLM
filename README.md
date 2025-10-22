# Unhackathon RAG Starter

A minimal retrieval-augmented generation (RAG) demo that builds a FAISS index over the `.txt` and `.pdf` files in `docs/`, retrieves the most relevant chunks, and asks a hosted LLM on Hugging Face Inference API to answer your question strictly from that context.

This branch is tuned for **local debugging on macOS** using a command-line workflow. There are no deployment instructions; everything runs on your machine while still calling the hosted Hugging Face Inference API for generation.

## Features
- CPU-only workflow with `sentence-transformers` embeddings (`intfloat/e5-small-v2`).
- Chunking (~180 words with overlap) and FAISS inner-product search.
- Command-line interface that cites sources and can display retrieved passages for inspection.
- Hosted generation via Hugging Face Inference API (`mistralai/Mistral-7B-Instruct-v0.3` by default) with automatic fallbacks for chat-only endpoints.

---

## 1. Local Setup (macOS)

### Prerequisites
- Python 3.10+
- Hugging Face Inference API token with access to the chosen model.

### Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
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

### Set Hugging Face Credentials
Export your token (and optionally choose a different model) in the same shell session:
```bash
export HF_TOKEN="hf_your_token_here"
export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"  # optional override
```
Check that the variables are set before running the CLI:
```bash
echo $HF_TOKEN
```

> **Note:** Some hosted models only expose a chat/completions endpoint. The app automatically retries with chat when necessary, but ensure the model you set in `HF_MODEL` is available through the Hugging Face Inference API.

### Run the CLI Debug Assistant
Ask a question inline by passing it as a positional argument:
```bash
python app.py "What does the topic document mention?"
```

Or enter the question interactively and inspect the retrieved context:
```bash
python app.py --show-context
```

The script prints the answer, source list, and (optionally) the retrieved passages so you can debug embedding, retrieval, and prompting issues without spinning up a UI. Use `--hf-model` or `--hf-token` to override the defaults directly from the command line when experimenting with different hosted models.

---

## Troubleshooting
- **"No .txt or .pdf files"**: Add at least one text file to `docs/` before running `build_index.py`.
- **"No chunks produced"**: PDF pages without extractable text may be scanned images; try providing a plain text file.
- **Weak or missing citations**: Re-run the query with a higher `--top-k` value or rebuild the index with a larger chunk size.
- **HF errors / 401**: Confirm that `HF_TOKEN` has access to the requested model.
- **404 when calling the model**: Check the spelling of `HF_MODEL` and make sure the model repo (e.g. `microsoft/Phi-3-mini-4k-instruct`) is public or shared with your token. Private or gated models require a token with permission.
- **"Model not supported for task text-generation"**: The app automatically retries using the chat completion API. If it persists, pick a chat-capable hosted model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) or verify the model's Inference API tasks.
- **Blank answers returned**: Some endpoints respond with structured payloads instead of plain strings. The CLI now normalises these responses, but if you still see an empty answer double-check the raw stderr output (`python app.py … 2>error.log`) for clues.
- **Gated models (e.g. Phi-3)**: Ensure you have accepted the model license on Hugging Face and that your `HF_TOKEN` has Inference API access. Try calling the model directly with a short prompt (`python - <<'PY' ...`) to confirm access before rerunning the RAG CLI.
- **Timeouts**: Use a smaller model via `HF_MODEL` or retry later; hosted inference endpoints can be busy.

Enjoy building on this RAG starter locally!
