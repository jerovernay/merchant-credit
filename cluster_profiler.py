# === cluster_profiler.py ===
"""Capa A del pipeline de ofertas: perfil agregado de cada cluster desde datos reales.

Responsabilidad: a partir del CSV de usuarios, calcular para cada cluster sus
intereses dominantes (autores, géneros), su intensidad de lectura y su ARPU
estimado. Todo determinístico — sin LLM. Esta señal alimenta al composer de
ofertas (offer_generator.py) para que el incentivo se ancle en lo que la gente
del cluster realmente lee, no en una noticia genérica.

La asignación de cluster (reglas sobre comportamiento) vive acá para que app.py
y offer_generator.py compartan la misma lógica.

Uso:
    from cluster_profiler import load_users, assign_clusters, build_cluster_profiles
    df = assign_clusters(load_users())
    perfiles = build_cluster_profiles(df)

    python cluster_profiler.py   # imprime los perfiles por consola
"""
from __future__ import annotations

from typing import Any

import pandas as pd

import config


def load_users() -> pd.DataFrame | None:
    """Carga el CSV de eventos sintéticos. Retorna None si no existe."""
    path = config.SYNTHETIC_DIR / "biblioteca_events.csv"
    if not path.exists():
        return None
    # decline_code debe ser string: "05" no debe leerse como int 5 (perdería el cero).
    return pd.read_csv(path, dtype={"decline_code": str})


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna cluster a cada usuario con reglas simples sobre comportamiento.

    Reglas (en orden de prioridad):
      - inactivo_reciente: > 45 días sin acceso
      - coleccionista: lista de deseos activa y <= 2 libros leídos en total
      - lector_voraz: >= 4 libros en 3 meses y apertura diaria
      - lector_fiel: >= 2 libros en 3 meses
      - lector_casual: el resto
    """
    def _assign(row: pd.Series) -> str:
        acceso = int(row.get("ultimo_acceso_dias", 30) or 30)
        libros_3m = int(row.get("libros_leidos_ultimos_3_meses", 0) or 0)
        libros_total = int(row.get("libros_leidos_total", 0) or 0)
        freq = str(row.get("frecuencia_apertura_app", "monthly")).lower()
        wishlist = bool(row.get("lista_deseos_activa", False))

        if acceso > 45:
            return "inactivo_reciente"
        if wishlist and libros_total <= 2:
            return "coleccionista"
        if libros_3m >= 4 and freq == "daily":
            return "lector_voraz"
        if libros_3m >= 2:
            return "lector_fiel"
        return "lector_casual"

    df = df.copy()
    df["cluster_id"] = df.apply(_assign, axis=1)
    return df


def _top_n(series: pd.Series, n: int = 3) -> list[dict[str, Any]]:
    """Top N valores no nulos de una serie, como lista de {valor, n}."""
    counts = series.dropna().astype(str)
    counts = counts[counts.str.strip() != ""]
    vc = counts.value_counts().head(n)
    return [{"valor": idx, "n": int(cnt)} for idx, cnt in vc.items()]


def _arpu_cluster(sub: pd.DataFrame) -> float:
    """ARPU del cluster como promedio ponderado del precio por tier de gasto."""
    if "tier_gasto" not in sub.columns or sub.empty:
        return float(config.ARPU_POR_TIER_ARS["Medium"])
    arpu = sub["tier_gasto"].map(config.ARPU_POR_TIER_ARS)
    arpu = arpu.dropna()
    if arpu.empty:
        return float(config.ARPU_POR_TIER_ARS["Medium"])
    return round(float(arpu.mean()), 0)


def build_cluster_profiles(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Construye el perfil agregado de cada cluster presente en el DataFrame.

    Returns:
        { cluster_id: {
            "usuarios":        int,
            "top_autores":     [{"valor": str, "n": int}, ...],
            "top_generos":     [{"valor": str, "n": int}, ...],
            "libros_mes_prom": float,
            "arpu_ars":        float,
            "tier_mix":        {"Low": int, "Medium": int, "High": int},
        } }
    """
    if "cluster_id" not in df.columns:
        df = assign_clusters(df)

    perfiles: dict[str, dict[str, Any]] = {}
    for cluster_id, sub in df.groupby("cluster_id"):
        libros_3m = sub.get("libros_leidos_ultimos_3_meses")
        libros_mes = round(float(libros_3m.mean()) / 3, 1) if libros_3m is not None else 0.0

        tier_mix = {}
        if "tier_gasto" in sub.columns:
            tier_mix = {k: int(v) for k, v in sub["tier_gasto"].value_counts().items()}

        perfiles[str(cluster_id)] = {
            "usuarios": int(len(sub)),
            "top_autores": _top_n(sub.get("autor_favorito", pd.Series(dtype=str)), 3),
            "top_generos": _top_n(sub.get("genero_favorito", pd.Series(dtype=str)), 3),
            "libros_mes_prom": libros_mes,
            "arpu_ars": _arpu_cluster(sub),
            "tier_mix": tier_mix,
        }
    return perfiles


if __name__ == "__main__":
    df = load_users()
    if df is None:
        print("No hay datos. Corré primero: python generator.py")
        raise SystemExit(1)

    df = assign_clusters(df)
    perfiles = build_cluster_profiles(df)

    print(f"\n--- Perfiles de cluster (N={len(df)}) ---\n")
    for cid, p in perfiles.items():
        autores = ", ".join(f"{a['valor']} ({a['n']})" for a in p["top_autores"]) or "-"
        generos = ", ".join(f"{g['valor']} ({g['n']})" for g in p["top_generos"]) or "-"
        print(f"[{cid}]  usuarios={p['usuarios']}  ARPU=ARS {p['arpu_ars']:.0f}  libros/mes={p['libros_mes_prom']}")
        print(f"    autores : {autores}")
        print(f"    generos : {generos}")
        print(f"    tier    : {p['tier_mix']}")
        print()
