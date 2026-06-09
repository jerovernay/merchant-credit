# === offer_matcher.py ===
"""Matcher determinístico: instancia una oferta CONCRETA por usuario desde el catálogo.

Es el núcleo anti-alucinación. El LLM nunca inventa un título, autor, evento o fecha:
acá el CÓDIGO selecciona un item real del inventario (data/catalogo.json) que pasó los
filtros, y solo eso se le ofrece al usuario.

Pipeline por usuario:
  1. Leer su historial -> autor/género que leyó ÚLTIMAMENTE (ponderado por recencia).
  2. Generar candidatos del catálogo según el incentivo elegido para su cluster.
  3. Filtrar: libro NO leído por el usuario · evento con fecha futura · (evento) misma ciudad.
  4. Escalera de confianza: individual (su historial) -> cluster (prior del segmento)
     -> genérico (sin señal). Nunca queda sin oferta; solo baja la especificidad.

Devuelve el texto listo para el slot {oferta_personal} + metadata para trazabilidad.

Uso:
    from offer_matcher import recomendar_oferta_personal
    rec = recomendar_oferta_personal(user_dict, incentivo_id="descuento_libro_autor",
                                     perfil_cluster=perfil)
"""
from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

import config

_CATALOGO: dict | None = None

# Vida media (días) para ponderar recencia al elegir el autor "de últimamente".
_RECENCIA_HALFLIFE_DIAS = 120
_RATING_IMPLICITO = 3.0


def _load_catalogo() -> dict:
    global _CATALOGO
    if _CATALOGO is None:
        with open(config.CATALOGO_PATH, encoding="utf-8") as f:
            _CATALOGO = json.load(f)
    return _CATALOGO


def _today() -> date:
    return date.fromisoformat(_load_catalogo()["_meta"]["fecha_referencia"])


def _parse_historial(user: dict[str, Any]) -> list[dict]:
    """Parsea el historial_libros (string JSON en el CSV) a lista de dicts."""
    raw = user.get("historial_libros")
    if isinstance(raw, list):
        return raw
    if not raw or not isinstance(raw, str):
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []


def _scores_recientes(historial: list[dict], today: date) -> tuple[dict[str, float], dict[str, float]]:
    """Puntaje por autor y por género ponderando rating * recencia (recientes pesan más)."""
    score_autor: dict[str, float] = {}
    score_genero: dict[str, float] = {}
    for b in historial:
        r = b.get("rating")
        r = float(r) if r is not None else _RATING_IMPLICITO
        try:
            dias = (today - date.fromisoformat(b["fecha_lectura"])).days
        except (ValueError, KeyError):
            dias = 365
        peso = r * math.exp(-dias / _RECENCIA_HALFLIFE_DIAS)
        if b.get("autor"):
            score_autor[b["autor"]] = score_autor.get(b["autor"], 0.0) + peso
        if b.get("genero"):
            score_genero[b["genero"]] = score_genero.get(b["genero"], 0.0) + peso
    return score_autor, score_genero


def _libros_no_leidos_por_autor(autor: str, leidos: set[str]) -> list[dict]:
    """Libros del catálogo de un autor que el usuario NO leyó, novedades primero."""
    libros = [b for b in _load_catalogo()["libros"]
              if b["autor"] == autor and b["titulo"] not in leidos]
    # Novedades primero; dentro de cada grupo, lanzamiento más reciente primero.
    return sorted(libros, key=lambda b: (b.get("es_novedad", False), b.get("fecha_lanzamiento", "")), reverse=True)


def _libros_no_leidos_por_genero(genero: str, leidos: set[str]) -> list[dict]:
    libros = [b for b in _load_catalogo()["libros"]
              if b["genero"] == genero and b["titulo"] not in leidos]
    return sorted(libros, key=lambda b: (b.get("es_novedad", False), b.get("fecha_lanzamiento", "")), reverse=True)


def _eventos_validos(autor: str, ciudad: str, today: date) -> tuple[list[dict], list[dict]]:
    """Eventos futuros del autor: (en la misma ciudad, en otra ciudad). Filtra los pasados."""
    futuros = [e for e in _load_catalogo()["eventos"]
               if e["autor"] == autor and date.fromisoformat(e["fecha"]) > today]
    misma = [e for e in futuros if str(e.get("ciudad", "")).lower() == str(ciudad or "").lower()]
    otra = [e for e in futuros if e not in misma]
    return misma, otra


def _fmt_fecha(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day:02d}/{d.month:02d}"


def _texto_libro(libro: dict, incentivo_id: str) -> str:
    """Frase para un libro según el tipo de incentivo."""
    base = f"«{libro['titulo']}», de {libro['autor']}"
    if incentivo_id == "descuento_libro_autor":
        inc = _load_incentivo(incentivo_id)
        pct = int(float(inc.get("descuento_pct", 0.2)) * 100)
        return f"{base}, con {pct}% off — todavía no lo leíste"
    if incentivo_id == "libro_regalo":
        # Regalo: la frase es solo el libro; el template ya dice "te regalamos ...".
        return base
    if incentivo_id == "acceso_anticipado":
        if libro.get("es_novedad"):
            return f"acceso anticipado a {base}, recién salido"
        return f"acceso anticipado a {base}"
    return base


def _load_incentivo(incentivo_id: str) -> dict:
    with open(config.INCENTIVOS_PATH, encoding="utf-8") as f:
        return json.load(f)["incentivos"].get(incentivo_id, {})


def recomendar_oferta_personal(
    user: dict[str, Any],
    incentivo_id: str,
    perfil_cluster: dict | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Instancia una oferta concreta del catálogo para un usuario.

    Args:
        user: fila del usuario (dict) con historial_libros, ciudad, etc.
        incentivo_id: incentivo elegido para el cluster (define qué tipo de item buscar).
        perfil_cluster: perfil agregado del cluster (prior para cold-start).
        today: fecha de referencia (default: la del catálogo).

    Returns:
        {
            "tipo_recomendacion": str,   # libro_descuento|libro_novedad|libro|evento|club|pack|mes_gratis|generico
            "confianza": str,            # individual | cluster | generico
            "autor": str | None,
            "genero": str | None,
            "item": dict | None,         # item del catálogo usado (trazabilidad)
            "texto": str,                # frase para el slot {oferta_personal}
        }
    """
    today = today or _today()
    historial = _parse_historial(user)
    leidos = {b["titulo"] for b in historial if b.get("titulo")}
    ciudad = user.get("ciudad", "")

    score_autor, score_genero = _scores_recientes(historial, today)
    autor = max(score_autor, key=score_autor.get) if score_autor else None
    genero = max(score_genero, key=score_genero.get) if score_genero else None
    confianza = "individual" if autor else ("cluster" if perfil_cluster else "generico")

    # Cold-start: caer al prior del cluster (autor/género dominante del segmento).
    if not autor and perfil_cluster:
        top_a = perfil_cluster.get("top_autores") or []
        top_g = perfil_cluster.get("top_generos") or []
        autor = top_a[0]["valor"] if top_a else None
        genero = top_g[0]["valor"] if top_g else None

    def _resultado(tipo: str, texto: str, item: dict | None) -> dict[str, Any]:
        return {"tipo_recomendacion": tipo, "confianza": confianza,
                "autor": autor, "genero": genero, "item": item, "texto": texto}

    # --- Incentivos que requieren un EVENTO concreto ---
    if incentivo_id == "entrada_evento" and autor:
        misma, otra = _eventos_validos(autor, ciudad, today)
        if misma:
            e = misma[0]
            return _resultado("evento", f"una entrada para «{e['titulo']}» de {e['autor']} "
                                         f"en {e['ciudad']}, el {_fmt_fecha(e['fecha'])}", e)
        if otra:
            e = otra[0]
            return _resultado("evento", f"una entrada para «{e['titulo']}» de {e['autor']} "
                                         f"({e['ciudad']}, {_fmt_fecha(e['fecha'])})", e)
        # No hay evento futuro que matchee -> degradar a un libro del autor.
        libros = _libros_no_leidos_por_autor(autor, leidos)
        if libros:
            return _resultado("libro", _texto_libro(libros[0], "acceso_anticipado"), libros[0])

    # --- Incentivos anclados en LIBRO de autor ---
    if incentivo_id in ("descuento_libro_autor", "acceso_anticipado", "libro_regalo") and autor:
        libros = _libros_no_leidos_por_autor(autor, leidos)
        if libros:
            tipo = {"descuento_libro_autor": "libro_descuento",
                    "libro_regalo": "libro_regalo"}.get(incentivo_id, "libro_novedad")
            return _resultado(tipo, _texto_libro(libros[0], incentivo_id), libros[0])
        # Leyó todo del autor -> recomendar por género.
        if genero:
            libros_g = _libros_no_leidos_por_genero(genero, leidos)
            if libros_g:
                return _resultado("libro", _texto_libro(libros_g[0], incentivo_id), libros_g[0])

    # --- Incentivos de club / pack (no necesitan item específico) ---
    if incentivo_id == "club_lectura_autor" and autor:
        return _resultado("club", f"el club de lectura de {autor}", None)
    if incentivo_id == "pack_genero_curado" and genero:
        return _resultado("pack", f"un pack curado de {genero} elegido por nuestro equipo", None)
    if incentivo_id == "mes_gratis":
        return _resultado("mes_gratis", "un mes de suscripción sin cargo al reactivar", None)

    # --- Fallbacks por confianza ---
    if autor:
        libros = _libros_no_leidos_por_autor(autor, leidos)
        if libros:
            return _resultado("libro", _texto_libro(libros[0], "acceso_anticipado"), libros[0])
    if genero:
        libros_g = _libros_no_leidos_por_genero(genero, leidos)
        if libros_g:
            return _resultado("libro", _texto_libro(libros_g[0], "acceso_anticipado"), libros_g[0])

    # Genérico total (cold-start sin prior de cluster).
    return _resultado("generico", "una selección de lecturas pensada para vos", None)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(config.SYNTHETIC_DIR / "biblioteca_events.csv")
    print("--- Pruebas del matcher (primeros usuarios con historial) ---\n")
    for inc in ["descuento_libro_autor", "entrada_evento", "acceso_anticipado"]:
        print(f"### incentivo: {inc}")
        mostrados = 0
        for _, row in df.iterrows():
            u = row.to_dict()
            rec = recomendar_oferta_personal(u, inc)
            print(f"  [{rec['confianza']:<10} {rec['tipo_recomendacion']:<15}] "
                  f"{u['first_name']} ({u['ciudad']}): {rec['texto']}")
            mostrados += 1
            if mostrados >= 4:
                break
        print()
