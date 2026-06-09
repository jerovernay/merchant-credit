# === domain_retriever.py ===
"""Recuperación de contexto de dominio por similaridad de embeddings.

Módulo de runtime: llamado por context_builder.py por cada evento.
Carga los embeddings pre-computados por crawler.py una sola vez y los
reutiliza en memoria durante toda la sesión.

Uso:
    from domain_retriever import retrieve_domain_context
    result = retrieve_domain_context(user_dict, top_k=2)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

CRAWLED_CONTEXT_PATH = Path("data/domain/crawled_context.json")
EMBED_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)

# Module-level cache: loaded once per process
_CHUNKS: list[dict[str, Any]] | None = None
_EMBEDDINGS_MATRIX: np.ndarray | None = None  # shape (N, 1536), L2-normalized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Inicializa cliente OpenAI síncrono desde variable de entorno."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Variable de entorno OPENAI_API_KEY no encontrada. "
            "Exportála antes de usar el retriever: export OPENAI_API_KEY=sk-..."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _load_crawled_context() -> bool:
    """Carga crawled_context.json en el cache de módulo. Retorna True si OK."""
    global _CHUNKS, _EMBEDDINGS_MATRIX

    if _CHUNKS is not None:
        return True

    if not CRAWLED_CONTEXT_PATH.exists():
        logger.warning(
            "Archivo de contexto no encontrado: %s. "
            "Corré crawler.py primero para generar los embeddings.",
            CRAWLED_CONTEXT_PATH,
        )
        return False

    try:
        with open(CRAWLED_CONTEXT_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Error al leer %s: %s", CRAWLED_CONTEXT_PATH, exc)
        return False

    chunks = data.get("chunks", [])
    if not chunks:
        logger.warning(
            "crawled_context.json está vacío. Corré crawler.py --force para regenerar."
        )
        return False

    # Build normalized embedding matrix for fast cosine similarity
    raw = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    _EMBEDDINGS_MATRIX = raw / norms
    _CHUNKS = chunks
    return True


def _build_profile_text(user_dict: dict) -> str:
    """Construye texto de perfil desde el dict de usuario."""
    genero = user_dict.get("genero_favorito") or ""
    autor = user_dict.get("autor_favorito") or ""
    frecuencia = user_dict.get("frecuencia_apertura_app") or ""
    libros = user_dict.get("libros_leidos_total", 0)
    resenas = user_dict.get("resenas_escritas", 0)
    return (
        f"{genero} {autor} hábitos de lectura {frecuencia} "
        f"libros leídos {libros} reseñas {resenas}"
    ).strip()


def _embed_text(client, text: str) -> np.ndarray | None:
    """Embeds a single text and returns an L2-normalized vector, or None on error."""
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as exc:
        logger.warning("Error al embeddear el perfil de usuario: %s", exc)
        return None


def _empty_result(profile_text: str) -> dict[str, Any]:
    return {
        "relevant_snippets": [],
        "profile_text_used": profile_text,
        "retrieval_ok": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_domain_context(user_dict: dict, top_k: int = 2) -> dict[str, Any]:
    """Recupera los top_k chunks más relevantes para el perfil del usuario.

    Args:
        user_dict: Dict con campos de comportamiento del usuario (genero_favorito,
                   autor_favorito, frecuencia_apertura_app, libros_leidos_total,
                   resenas_escritas). Campos faltantes se ignoran.
        top_k: Cantidad de chunks a retornar.

    Returns:
        {
            "relevant_snippets": [{"url", "tags", "text", "similarity_score"}],
            "profile_text_used": str,
            "retrieval_ok": bool,
        }
    """
    profile_text = _build_profile_text(user_dict)

    if not _load_crawled_context():
        return _empty_result(profile_text)

    try:
        client = _get_openai_client()
    except ValueError as exc:
        logger.warning(str(exc))
        return _empty_result(profile_text)

    profile_vec = _embed_text(client, profile_text)
    if profile_vec is None:
        return _empty_result(profile_text)

    # Cosine similarity: dot product of L2-normalized vectors
    scores = _EMBEDDINGS_MATRIX @ profile_vec  # shape (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]

    snippets = []
    for idx in top_indices:
        chunk = _CHUNKS[idx]
        snippets.append({
            "url": chunk["url"],
            "tags": chunk.get("tags", []),
            "text": chunk["text"],
            "similarity_score": round(float(scores[idx]), 4),
        })

    return {
        "relevant_snippets": snippets,
        "profile_text_used": profile_text,
        "retrieval_ok": True,
    }
