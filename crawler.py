# === crawler.py ===
"""Crawl offline de URLs del dominio literario argentino.

Cron job offline: corre una vez, guarda embeddings en disco.
NO se llama durante el procesamiento de eventos en tiempo real.

Uso:
    python crawler.py              # skip si ya existe crawled_context.json
    python crawler.py --force      # fuerza re-crawl aunque ya exista
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

URLS_PATH = Path("data/crawler/urls.json")
OUTPUT_PATH = Path("data/domain/crawled_context.json")

KEYWORDS = {"suscripción", "lectura", "libro", "lector", "género", "autor", "hábito"}
CHUNK_MAX_WORDS = 400
CHUNK_MIN_WORDS = 50
EMBED_BATCH_SIZE = 20
EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Config / API client
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Inicializa cliente OpenAI desde variable de entorno."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Variable de entorno OPENAI_API_KEY no encontrada. "
            "Exportála antes de correr el crawler: export OPENAI_API_KEY=sk-..."
        )
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=api_key)


def _load_urls() -> list[dict[str, Any]]:
    """Carga lista de URLs desde data/crawler/urls.json."""
    with open(URLS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Divide texto en chunks por párrafo, max CHUNK_MAX_WORDS por chunk.

    Descarta chunks que no contienen ninguna keyword del dominio.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > CHUNK_MAX_WORDS and current_words:
            chunks.append(" ".join(current_words))
            current_words = para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    # Filter by min length and keyword presence
    filtered = []
    for chunk in chunks:
        if len(chunk.split()) < CHUNK_MIN_WORDS:
            continue
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in KEYWORDS):
            filtered.append(chunk)

    return filtered


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts via OpenAI API."""
    response = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


async def _embed_all(client, texts: list[str]) -> list[list[float]]:
    """Embeds all texts in batches of EMBED_BATCH_SIZE."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_embeddings = await _embed_batch(client, batch)
        embeddings.extend(batch_embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

async def _crawl_url(url: str) -> str | None:
    """Crawls a single URL and returns markdown content, or None on failure."""
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

        config = CrawlerRunConfig(
            word_count_threshold=50,
            exclude_external_links=True,
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if result.success and result.markdown:
                return result.markdown
            print(f"    [!] Sin contenido en {url}")
            return None
    except Exception as exc:
        print(f"    [!] Error al crawlear {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def crawl_and_embed(force: bool = False) -> None:
    """Crawlea URLs, genera embeddings y guarda en data/domain/crawled_context.json.

    Args:
        force: Si True, re-crawlea aunque el archivo de salida ya exista.
    """
    if OUTPUT_PATH.exists() and not force:
        print(f"Ya existe {OUTPUT_PATH}, usá force=True para re-crawlear.")
        return

    client = _get_openai_client()
    urls_data = _load_urls()
    total = len(urls_data)

    all_chunks: list[dict[str, Any]] = []
    all_texts: list[str] = []
    chunk_meta: list[dict[str, Any]] = []  # url + tags per chunk, parallel to all_texts

    # --- Crawl phase ---
    for idx, entry in enumerate(urls_data, start=1):
        url = entry["url"]
        tags = entry.get("tags", [])
        print(f"Procesando URL {idx}/{total}: {url}")

        markdown = await _crawl_url(url)
        if markdown is None:
            continue

        chunks = _chunk_text(markdown)
        print(f"    → {len(chunks)} chunks válidos extraídos")

        for chunk in chunks:
            all_texts.append(chunk)
            chunk_meta.append({"url": url, "tags": tags})

    if not all_texts:
        print("Sin chunks para embeddear. Verificá las URLs y los keywords.")
        return

    # --- Embed phase ---
    print(f"\nEmbeddeando {len(all_texts)} chunks en batches de {EMBED_BATCH_SIZE}...")
    try:
        embeddings = await _embed_all(client, all_texts)
    except Exception as exc:
        print(f"[!] Error en embedding: {exc}")
        raise

    for i, (text, meta, embedding) in enumerate(zip(all_texts, chunk_meta, embeddings)):
        all_chunks.append({
            "url": meta["url"],
            "tags": meta["tags"],
            "text": text,
            "embedding": embedding,
        })

    # --- Save phase ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado: {OUTPUT_PATH} ({len(all_chunks)} chunks, {len(all_texts)} embeddings)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawler offline de contexto literario argentino."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-crawlea aunque crawled_context.json ya exista.",
    )
    args = parser.parse_args()
    asyncio.run(crawl_and_embed(force=args.force))


if __name__ == "__main__":
    main()
