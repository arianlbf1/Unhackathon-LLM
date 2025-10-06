import os, json
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

INDEX_DIR   = "index"
EMBED_MODEL = "intfloat/e5-small-v2"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You answer ONLY using the provided CONTEXT. "
    "If the answer is not present, say: 'I don't know based on the provided sources.' "
    "Cite sources as [#] matching the order of the context items."
)

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_MODEL = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)

st.set_page_config(page_title="Unhackathon Q&A", page_icon="❓")
st.title("Unhackathon Q&A (RAG)")

if HF_TOKEN is None:
    st.warning("HF_TOKEN is not set. Set it as an environment variable or a Cloud Secret.")

@st.cache_resource
def load_assets():
    idx = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    emb = SentenceTransformer(EMBED_MODEL)
    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN, timeout=60)
    return idx, chunks, emb, client

def embed_query(emb, q: str):
    v = emb.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def retrieve(idx, chunks, emb, q: str, k: int = 4):
    qv = embed_query(emb, q)
    scores, ids = idx.search(qv, k)
    picks = ids[0].tolist()
    ctx_texts, labels = [], []
    for rank, i in enumerate(picks, start=1):
        c = chunks[i]
        ctx_texts.append(c["text"])
        label = f"[{rank}] {c['source']}" + (f" p.{c['page']}" if c['page'] else "")
        labels.append(label)
    return ctx_texts, labels

def make_prompt(question: str, ctx_texts: list[str]) -> str:
    ctx_block = "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(ctx_texts))
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nCONTEXT:\n{ctx_block}\n\nAnswer:"

def generate_answer(client: InferenceClient, prompt: str) -> str:
    out = client.text_generation(
        prompt,
        max_new_tokens=220,
        temperature=0.2,
        do_sample=False,
        return_full_text=False,
    )
    return out.strip()

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Top-K passages", 2, 8, 4, 1)
    st.caption("If citations feel weak, increase Top-K.")

question = st.text_input("Ask a question about your docs:")
if st.button("Answer") and question.strip():
    try:
        idx, chunks, emb, client = load_assets()
        ctx_texts, labels = retrieve(idx, chunks, emb, question, k=k)
        if not ctx_texts:
            st.write("I don't know based on the provided sources.")
        else:
            prompt = make_prompt(question, ctx_texts)
            answer = generate_answer(client, prompt)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Sources")
            for s in labels:
                st.write(f"- {s}")
            with st.expander("Show retrieved passages"):
                for txt, lab in zip(ctx_texts, labels):
                    st.markdown(f"**{lab}**")
                    st.write(txt)
    except Exception as e:
        st.error(f"Error: {e}")
