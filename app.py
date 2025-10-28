"""Command-line interface for asking questions against the local FAISS index."""

import argparse
import json
import os
import sys
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMBED_MODEL = "intfloat/e5-small-v2"
DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = (
    "You answer ONLY using the provided CONTEXT. "
    "If the answer is not present, say: 'I don't know based on the provided sources.' "
    "Cite sources as [#] matching the order of the context items."
)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)


class MissingCredentialError(RuntimeError):
    """Raised when the HF token is missing."""


@lru_cache(maxsize=1)
def load_assets(
    index_dir: str,
    embed_model: str,
    hf_model: str,
    hf_token: str,
) -> Tuple[faiss.Index, List[dict], SentenceTransformer, InferenceClient]:
    if not hf_token:
        raise MissingCredentialError(
            "HF_TOKEN is required. Set it as an environment variable or pass --hf-token."
        )

    index_path = os.path.join(index_dir, "faiss.index")
    chunks_path = os.path.join(index_dir, "chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing FAISS index at {index_path}. Build it via `python build_index.py`."
        )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Missing chunk metadata at {chunks_path}. Rebuild with `python build_index.py`."
        )

    idx = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    emb = SentenceTransformer(embed_model)
    client = InferenceClient(model=hf_model, token=hf_token, timeout=60)
    return idx, chunks, emb, client


def embed_query(emb: SentenceTransformer, q: str) -> np.ndarray:
    v = emb.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")


def retrieve(
    idx: faiss.Index, chunks: Sequence[dict], emb: SentenceTransformer, q: str, k: int = 4
) -> Tuple[List[str], List[str]]:
    qv = embed_query(emb, q)
    scores, ids = idx.search(qv, k)
    picks = [pid for pid in ids[0].tolist() if 0 <= pid < len(chunks)]
    ctx_texts, labels = [], []
    for rank, i in enumerate(picks, start=1):
        c = chunks[i]
        ctx_texts.append(c["text"])
        label = f"[{rank}] {c['source']}" + (f" p.{c['page']}" if c['page'] else "")
        labels.append(label)
    return ctx_texts, labels


def build_context_block(ctx_texts: List[str]) -> str:
    return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(ctx_texts))


def make_prompt(question: str, ctx_block: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nCONTEXT:\n{ctx_block}\n\nAnswer:"
    )


def _extract_error_snippet(err: HTTPError) -> str:
    snippet = str(err) or ""
    response = getattr(err, "response", None)
    payload = None
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message") or payload.get("detail")
            if isinstance(detail, str):
                return detail
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            snippet = text.strip()
    return snippet


def _is_task_mismatch_error(err: HTTPError) -> bool:
    snippet = _extract_error_snippet(err)
    if isinstance(snippet, str) and "not supported for task" in snippet.lower():
        return True
    text = str(err)
    return isinstance(text, str) and "not supported for task" in text.lower()


def _coerce_chat_content(output) -> str:
    choices = getattr(output, "choices", None)
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        content = None
        if message is not None:
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
        if not content and isinstance(first, dict):
            content = first.get("content")
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        if isinstance(content, str) and content.strip():
            return content.strip()
    if isinstance(output, dict):
        dict_choices = output.get("choices")
        if isinstance(dict_choices, list) and dict_choices:
            first = dict_choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                content = first.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    generated = getattr(output, "generated_text", None)
    if isinstance(generated, str) and generated.strip():
        return generated.strip()
    generated_list = getattr(output, "generated_texts", None)
    if isinstance(generated_list, Iterable):
        for item in generated_list:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                txt = item.get("text") or item.get("generated_text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    if isinstance(output, list):
        for entry in output:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
            if isinstance(entry, dict):
                txt = entry.get("content") or entry.get("generated_text") or entry.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    return ""


def _call_chat_completion(
    client: InferenceClient, question: str, ctx_block: str
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nCONTEXT:\n{ctx_block}\n\nAnswer:",
        },
    ]
    output = client.chat_completion(
        messages=messages,
        max_tokens=220,
        temperature=0.2,
    )
    content = _coerce_chat_content(output)
    if not content:
        raise ValueError("Empty response from chat completion.")
    return content


def _coerce_text_generation_output(output) -> str:
    if isinstance(output, str):
        return output.strip()
    generated = getattr(output, "generated_text", None)
    if isinstance(generated, str) and generated.strip():
        return generated.strip()
    generated_list = getattr(output, "generated_texts", None)
    if isinstance(generated_list, Iterable):
        for item in generated_list:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                txt = item.get("text") or item.get("generated_text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    if isinstance(output, dict):
        txt = output.get("generated_text") or output.get("content") or output.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        generated_list = output.get("generated_texts")
        if isinstance(generated_list, Iterable):
            for item in generated_list:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("generated_text")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    if isinstance(output, list):
        for entry in output:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
            if isinstance(entry, dict):
                txt = entry.get("generated_text") or entry.get("text") or entry.get("content")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    return ""


def generate_answer(
    client: InferenceClient, question: str, ctx_texts: List[str]
) -> str:
    ctx_block = build_context_block(ctx_texts)
    prompt = make_prompt(question, ctx_block)
    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=220,
            temperature=0.2,
            do_sample=False,
            return_full_text=False,
        )
        coerced = _coerce_text_generation_output(out)
        if not coerced:
            raise ValueError("Empty response from text-generation endpoint.")
        return coerced
    except HTTPError as err:
        if not _is_task_mismatch_error(err):
            raise
        chat_answer = _call_chat_completion(client, question, ctx_block)
        return chat_answer.strip()


def _format_hf_http_error(err: HTTPError) -> Tuple[str, str | None]:
    """Return a concise error message and optional hint for HTTP errors."""
    response = getattr(err, "response", None)
    base = str(err).strip() or "Hugging Face Inference API request failed."
    detail = None
    status = None

    if response is not None:
        status = getattr(response, "status_code", None)
        try:
            payload = response.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message") or payload.get("detail")
        elif payload is None and hasattr(response, "text"):
            text = response.text.strip()
            if text and text != base:
                detail = text

    hint = None
    if status == 404:
        hint = (
            "Model not found. Double-check the `HF_MODEL` environment variable or ensure "
            "the model repo is public and accessible with your token."
        )
    elif status in {401, 403}:
        hint = (
            "Authentication failed. Verify `HF_TOKEN` is set and has access to the model "
            "(use a token with Inference API permissions)."
        )
    elif status == 400 and _is_task_mismatch_error(err):
        hint = (
            "The selected model only exposes a chat/completions endpoint. The app will "
            "retry using chat, but ensure the model supports inference via the hosted API."
        )
    elif status == 429:
        hint = (
            "Rate limit hit. Wait a moment or switch to a smaller/less busy model."
        )
    elif status in {500, 502, 503}:
        hint = (
            "The hosted inference endpoint is unavailable. Retry shortly or choose a "
            "different model via `HF_MODEL`."
        )

    message = base if detail is None else f"{base} — {detail}"
    return message, hint


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the local FAISS index and ask a hosted Hugging Face model."
    )
    parser.add_argument(
        "question_tokens",
        nargs="*",
        help="Question to ask. If omitted, the script prompts via stdin.",
    )
    parser.add_argument(
        "-q",
        "--question",
        dest="question_text",
        help="Provide the question explicitly without relying on positional arguments.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved context passages for inspection.",
    )
    parser.add_argument(
        "--hf-model",
        help="Override the Hugging Face model ID (defaults to env HF_MODEL or configured default).",
    )
    parser.add_argument(
        "--hf-token",
        help="Override the Hugging Face token (defaults to env HF_TOKEN).",
    )
    parser.add_argument(
        "--index-dir",
        default=INDEX_DIR,
        help="Directory containing faiss.index and chunks.json (default: index).",
    )
    return parser


def _prompt_for_question() -> str:
    try:
        return input("Enter your question: ").strip()
    except EOFError:
        return ""


def _print_sources(labels: List[str]) -> None:
    if not labels:
        print("No sources retrieved.")
        return
    print("Sources:")
    for label in labels:
        print(f" - {label}")


def _print_context(ctx_texts: List[str], labels: List[str]) -> None:
    if not ctx_texts:
        return
    print("\n--- Retrieved Passages ---")
    for label, text in zip(labels, ctx_texts):
        print(label)
        print(text)
        print()


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    question = (args.question_text or " ".join(args.question_tokens)).strip()
    if not question:
        question = _prompt_for_question()

    if not question:
        print("Provide a question via CLI arguments or stdin.", file=sys.stderr)
        return 1

    hf_token = args.hf_token or HF_TOKEN
    hf_model = args.hf_model or HF_MODEL

    try:
        idx, chunks, emb, client = load_assets(args.index_dir, EMBED_MODEL, hf_model, hf_token)
    except MissingCredentialError as err:
        print(err, file=sys.stderr)
        return 2
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 3
    except Exception as err:  # pragma: no cover - defensive catch for asset loading
        print(f"Failed to load assets: {err}", file=sys.stderr)
        return 4

    ctx_texts, labels = retrieve(idx, chunks, emb, question, k=args.top_k)

    if not ctx_texts:
        print("I don't know based on the provided sources.")
        return 0

    try:
        answer = generate_answer(client, question, ctx_texts)
    except HTTPError as err:
        message, hint = _format_hf_http_error(err)
        print(message, file=sys.stderr)
        if hint:
            print(f"Hint: {hint}", file=sys.stderr)
        return 5
    except Exception as err:
        print(f"Generation failed: {err}", file=sys.stderr)
        return 6

    print("Answer:\n")
    print(answer)
    print()
    _print_sources(labels)

    if args.show_context:
        _print_context(ctx_texts, labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
