# === clustering.py ===
"""Clustering module for Vertical 1: Biblioteca de suscripción.

Assigns user type labels (lector_voraz, lector_fiel, etc.) based on behavioral features
using KMeans clustering. Features are normalized and cluster → label mapping is
configurable via data/services/biblioteca.json.

Uso:
    python clustering.py --train              # Train on default dataset
    python clustering.py --predict-demo       # Test predict() on a sample user

    from clustering import train, predict
    train(df)
    result = predict({"libros_leidos_total": 15, ...})
"""
from __future__ import annotations

import argparse
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config

# Behavioral features for clustering (exact order for reproducibility)
BEHAVIORAL_FEATURES = [
    "libros_leidos_total",
    "libros_leidos_ultimos_3_meses",
    "rating_promedio_dado",
    "frecuencia_apertura_app_num",
    "ultimo_acceso_dias",
    "lista_deseos_activa",
    "resenas_escritas",
]

# Encoding for categorical frecuencia_apertura_app
FRECUENCIA_APERTURA_MAP = {
    "daily": 3,
    "weekly": 2,
    "monthly": 1,
}

# Module-level cache for models
_MODEL: KMeans | None = None
_SCALER: StandardScaler | None = None
_CLUSTER_LABELS: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_biblioteca_config() -> dict:
    """Carga configuración de Vertical 1."""
    with open(config.BIBLIOTECA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_cluster_labels() -> dict[str, str]:
    """Lee el mapeo cluster_index → user_type desde biblioteca.json."""
    global _CLUSTER_LABELS
    if _CLUSTER_LABELS is None:
        bib_config = _load_biblioteca_config()
        _CLUSTER_LABELS = bib_config.get("cluster_labels", {})
        # Ensure all cluster indices are covered (with defaults if missing)
        for i in range(config.KMEANS_N_CLUSTERS):
            if str(i) not in _CLUSTER_LABELS:
                _CLUSTER_LABELS[str(i)] = "lector_casual"
    return _CLUSTER_LABELS


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _encode_features(user_dict: dict) -> dict:
    """Encodes categorical features and ensures all required fields are present.

    Converts frecuencia_apertura_app (str) → frecuencia_apertura_app_num (int).
    """
    encoded = user_dict.copy()

    freq_str = str(user_dict.get("frecuencia_apertura_app", "monthly")).lower()
    encoded["frecuencia_apertura_app_num"] = FRECUENCIA_APERTURA_MAP.get(freq_str, 1)

    # Ensure bool → int conversion for lista_deseos_activa
    encoded["lista_deseos_activa"] = int(bool(user_dict.get("lista_deseos_activa", False)))

    return encoded


def _extract_feature_vector(user_dict: dict) -> np.ndarray:
    """Extracts and orders behavioral features from user dict into a feature vector."""
    encoded = _encode_features(user_dict)
    features = []
    for feat_name in BEHAVIORAL_FEATURES:
        val = encoded.get(feat_name, 0)
        features.append(float(val))
    return np.array(features, dtype=np.float32).reshape(1, -1)


def _extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extracts all behavioral features from DataFrame."""
    freq_mapped = df.get("frecuencia_apertura_app", pd.Series(["monthly"] * len(df)))
    freq_num = freq_mapped.map(lambda x: FRECUENCIA_APERTURA_MAP.get(str(x).lower(), 1))

    feature_df = pd.DataFrame({
        "libros_leidos_total": pd.to_numeric(df.get("libros_leidos_total", 0), errors="coerce").fillna(0),
        "libros_leidos_ultimos_3_meses": pd.to_numeric(df.get("libros_leidos_ultimos_3_meses", 0), errors="coerce").fillna(0),
        "rating_promedio_dado": pd.to_numeric(df.get("rating_promedio_dado", 0.0), errors="coerce").fillna(0.0),
        "frecuencia_apertura_app_num": freq_num.fillna(1),
        "ultimo_acceso_dias": pd.to_numeric(df.get("ultimo_acceso_dias", 30), errors="coerce").fillna(30),
        "lista_deseos_activa": df.get("lista_deseos_activa", False).astype(int),
        "resenas_escritas": pd.to_numeric(df.get("resenas_escritas", 0), errors="coerce").fillna(0),
    })
    return feature_df.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _load_model() -> tuple[KMeans | None, StandardScaler | None]:
    """Loads KMeans model and scaler from disk if available."""
    global _MODEL, _SCALER

    if _MODEL is not None and _SCALER is not None:
        return _MODEL, _SCALER

    model_path = config.KMEANS_BIBLIOTECA_PATH
    scaler_path = config.KMEANS_SCALER_PATH

    try:
        if model_path.exists() and scaler_path.exists():
            with open(model_path, "rb") as f:
                _MODEL = pickle.load(f)
            with open(scaler_path, "rb") as f:
                _SCALER = pickle.load(f)
            return _MODEL, _SCALER
    except Exception:
        pass

    return None, None


def _save_model(model: KMeans, scaler: StandardScaler) -> None:
    """Saves model and scaler to disk."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.KMEANS_BIBLIOTECA_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(config.KMEANS_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> None:
    """Fits KMeans on behavioral features and saves model + scaler.

    Args:
        df: DataFrame with all BEHAVIORAL_FEATURES columns.

    Prints cluster sizes and top 2 features per cluster for validation.
    """
    X = _extract_feature_matrix(df)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    model = KMeans(
        n_clusters=config.KMEANS_N_CLUSTERS,
        random_state=42,
        n_init=10,
    )
    model.fit(X_scaled)

    # Save model and scaler
    _save_model(model, scaler)

    # Print diagnostics
    labels = model.labels_
    cluster_labels = _get_cluster_labels()

    print(f"\n--- KMeans Training Complete (n_clusters={config.KMEANS_N_CLUSTERS}) ---")
    print(f"  Model saved: {config.KMEANS_BIBLIOTECA_PATH}")
    print(f"  Scaler saved: {config.KMEANS_SCALER_PATH}\n")

    for cluster_idx in range(config.KMEANS_N_CLUSTERS):
        mask = labels == cluster_idx
        cluster_size = mask.sum()
        user_type = cluster_labels.get(str(cluster_idx), "lector_casual")

        # Top 2 features for this cluster (highest variance in this cluster)
        cluster_data = X_scaled[mask]
        feature_variance = cluster_data.std(axis=0)
        top_feat_indices = np.argsort(feature_variance)[-2:][::-1]
        top_features = [
            BEHAVIORAL_FEATURES[i] for i in top_feat_indices
        ]

        print(
            f"  Cluster {cluster_idx} [{user_type}]: {cluster_size} users | "
            f"Top features: {', '.join(top_features)}"
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(user_dict: dict) -> dict[str, Any]:
    """Assigns user type and confidence to a single user.

    Args:
        user_dict: Dict with behavioral feature keys (can be sparse; missing
                  values default to 0).

    Returns:
        {
            "user_type": str,        # label from cluster_labels mapping
            "confidence": float,     # 1 / (1 + distance to centroid)
        }
    """
    model, scaler = _load_model()

    # Fallback if model not trained
    if model is None or scaler is None:
        return {
            "user_type": "lector_casual",
            "confidence": 0.0,
        }

    # Extract and scale features
    X = _extract_feature_vector(user_dict)
    X_scaled = scaler.transform(X).astype(np.float32)

    # Predict cluster
    cluster_idx = model.predict(X_scaled)[0]

    # Calculate distance to centroid
    centroid = model.cluster_centers_[cluster_idx]
    distance = np.linalg.norm(X_scaled[0] - centroid)
    confidence = 1.0 / (1.0 + distance)

    # Map cluster to label
    cluster_labels = _get_cluster_labels()
    user_type = cluster_labels.get(str(cluster_idx), "lector_casual")

    return {
        "user_type": user_type,
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo_predict() -> None:
    """Predicts on a sample user for demonstration."""
    sample_user = {
        "libros_leidos_total": 25,
        "libros_leidos_ultimos_3_meses": 4,
        "rating_promedio_dado": 4.2,
        "frecuencia_apertura_app": "daily",
        "ultimo_acceso_dias": 2,
        "lista_deseos_activa": True,
        "resenas_escritas": 8,
    }

    result = predict(sample_user)
    print(f"\n--- Prediction Demo ---")
    print(f"  Sample user: {sample_user}")
    print(f"  Result: {result}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clustering module for Vertical 1 (Biblioteca)."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train KMeans on dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config.DATA_DIR / "synthetic" / "biblioteca_events.csv"),
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--predict-demo",
        action="store_true",
        help="Test predict() on a sample user.",
    )

    args = parser.parse_args()

    if args.train:
        df = pd.read_csv(args.dataset)
        train(df)
    elif args.predict_demo:
        _demo_predict()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
