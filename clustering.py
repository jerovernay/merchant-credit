# === clustering.py ===
"""Clustering de tipos de usuario para Vertical 1: Biblioteca de suscripción.

Entrena un KMeans sobre los features comportamentales del lector y asigna
la etiqueta ``user_type`` (lector_voraz, lector_fiel, etc.) a cada usuario.

Diseño:
  - KMeans con n_clusters = config.KMEANS_N_CLUSTERS (default 5).
  - Features normalizados con StandardScaler (persistido junto al modelo).
  - El mapa cluster_id → user_type vive en data/services/biblioteca.json.
  - Confianza = 1 / (1 + distancia_al_centroide) ∈ (0, 1].

Uso:
    python clustering.py --train
    python clustering.py --predict-demo
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Feature list (must match behavioral_features in biblioteca.json)
# ---------------------------------------------------------------------------

BEHAVIORAL_FEATURES: list[str] = [
    "libros_leidos_total",
    "libros_leidos_ultimos_3_meses",
    "rating_promedio_dado",
    "frecuencia_apertura_app_num",  # daily=3, weekly=2, monthly=1
    "ultimo_acceso_dias",
    "lista_deseos_activa",
    "resenas_escritas",
    "tenure_meses",
    "prior_recoveries",
]

_FRECUENCIA_MAP: dict[str, int] = {"daily": 3, "weekly": 2, "monthly": 1}


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_biblioteca_config() -> dict[str, Any]:
    """Carga la configuración del vertical biblioteca."""
    with open(config.BIBLIOTECA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_cluster_label_map() -> dict[str, str]:
    """Devuelve el mapa cluster_id (str) → user_type desde biblioteca.json."""
    cfg = _load_biblioteca_config()
    return cfg["cluster_label_map"]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _prepare_features(user_dict: dict) -> np.ndarray:
    """Convierte un dict de usuario en vector de features normalizable."""
    freq_raw = str(user_dict.get("frecuencia_apertura_app", "monthly")).lower()
    freq_num = _FRECUENCIA_MAP.get(freq_raw, 1)

    row = [
        float(user_dict.get("libros_leidos_total", 0)),
        float(user_dict.get("libros_leidos_ultimos_3_meses", 0)),
        float(user_dict.get("rating_promedio_dado", 0.0)),
        float(freq_num),
        float(user_dict.get("ultimo_acceso_dias", 30)),
        float(int(bool(user_dict.get("lista_deseos_activa", False)))),
        float(user_dict.get("resenas_escritas", 0)),
        float(user_dict.get("tenure_meses", 1)),
        float(user_dict.get("prior_recoveries", 0)),
    ]
    return np.array(row, dtype=np.float32)


def _prepare_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Prepara la matriz de features desde un DataFrame de usuarios."""
    freq_mapped = df.get("frecuencia_apertura_app", pd.Series(["monthly"] * len(df)))
    freq_num = freq_mapped.map(lambda x: _FRECUENCIA_MAP.get(str(x).lower(), 1))

    feature_df = pd.DataFrame({
        "libros_leidos_total": pd.to_numeric(df.get("libros_leidos_total", 0), errors="coerce").fillna(0),
        "libros_leidos_ultimos_3_meses": pd.to_numeric(df.get("libros_leidos_ultimos_3_meses", 0), errors="coerce").fillna(0),
        "rating_promedio_dado": pd.to_numeric(df.get("rating_promedio_dado", 0.0), errors="coerce").fillna(0.0),
        "frecuencia_apertura_app_num": freq_num.fillna(1),
        "ultimo_acceso_dias": pd.to_numeric(df.get("ultimo_acceso_dias", 30), errors="coerce").fillna(30),
        "lista_deseos_activa": df.get("lista_deseos_activa", False).astype(int),
        "resenas_escritas": pd.to_numeric(df.get("resenas_escritas", 0), errors="coerce").fillna(0),
        "tenure_meses": pd.to_numeric(df.get("tenure_meses", 1), errors="coerce").fillna(1),
        "prior_recoveries": pd.to_numeric(df.get("prior_recoveries", 0), errors="coerce").fillna(0),
    })
    return feature_df.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> Any:
    """Entrena KMeans sobre el DataFrame de usuarios y persiste modelo + scaler.

    Args:
        df: DataFrame con al menos las columnas de BEHAVIORAL_FEATURES.

    Returns:
        El objeto KMeans entrenado.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"[clustering] Entrenando KMeans con {len(df)} filas, k={config.KMEANS_N_CLUSTERS}.")

    X = _prepare_dataframe(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=config.KMEANS_N_CLUSTERS,
        random_state=42,
        n_init=10,
    )
    kmeans.fit(X_scaled)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.KMEANS_BIBLIOTECA_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    with open(config.KMEANS_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    inertia = kmeans.inertia_
    print(f"[clustering] Entrenamiento completo. Inercia: {inertia:.2f}")
    print(f"[clustering] Modelo guardado en {config.KMEANS_BIBLIOTECA_PATH}")
    print(f"[clustering] Scaler guardado en {config.KMEANS_SCALER_PATH}")
    return kmeans


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_KMEANS_CACHE: tuple | None = None


def _load_model() -> tuple[Any, Any]:
    """Carga KMeans + scaler desde disco (lazy singleton)."""
    global _KMEANS_CACHE
    if _KMEANS_CACHE is None:
        if not config.KMEANS_BIBLIOTECA_PATH.exists():
            raise FileNotFoundError(
                "Modelo de clustering no entrenado. Ejecutá: python clustering.py --train"
            )
        with open(config.KMEANS_BIBLIOTECA_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(config.KMEANS_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        _KMEANS_CACHE = (kmeans, scaler)
    return _KMEANS_CACHE


def predict(user_dict: dict) -> dict[str, Any]:
    """Asigna user_type y confianza a un usuario a partir de su comportamiento.

    Args:
        user_dict: Dict con campos comportamentales del usuario (signup + in-app).

    Returns:
        {"user_type": str, "confidence": float}
    """
    try:
        kmeans, scaler = _load_model()
    except FileNotFoundError:
        return {"user_type": "lector_casual", "confidence": 0.0}

    label_map = _get_cluster_label_map()

    x = _prepare_features(user_dict).reshape(1, -1)
    x_scaled = scaler.transform(x)

    cluster_id = int(kmeans.predict(x_scaled)[0])
    user_type = label_map.get(str(cluster_id), "lector_casual")

    # Confidence: inverse distance to assigned centroid, normalized to (0, 1].
    centroid = kmeans.cluster_centers_[cluster_id]
    dist = float(np.linalg.norm(x_scaled[0] - centroid))
    confidence = round(1.0 / (1.0 + dist), 4)

    return {"user_type": user_type, "confidence": confidence}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada CLI para entrenar o hacer una predicción de demo."""
    parser = argparse.ArgumentParser(description="Clustering de usuarios — Vertical Biblioteca.")
    parser.add_argument("--train", action="store_true", help="Entrena el modelo KMeans.")
    parser.add_argument("--dataset", type=str, default=str(config.DATASET_PATH),
                        help="Ruta al CSV de entrenamiento.")
    parser.add_argument("--predict-demo", action="store_true",
                        help="Predicción de demo con un usuario sintético.")
    args = parser.parse_args()

    if args.train:
        path = Path(args.dataset)
        if not path.exists():
            print(f"[clustering] Error: dataset no encontrado en {path}. Ejecutá python generator.py primero.")
            return
        df = pd.read_csv(path)
        train(df)

    if args.predict_demo:
        demo_user = {
            "libros_leidos_total": 35,
            "libros_leidos_ultimos_3_meses": 6,
            "rating_promedio_dado": 4.2,
            "frecuencia_apertura_app": "daily",
            "ultimo_acceso_dias": 2,
            "lista_deseos_activa": True,
            "resenas_escritas": 8,
            "tenure_meses": 18,
            "prior_recoveries": 1,
        }
        result = predict(demo_user)
        print("\n--- Demo de clustering ---")
        print(f"  user_type : {result['user_type']}")
        print(f"  confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
