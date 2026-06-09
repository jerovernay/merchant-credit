"""Modelo de scoring y selección de acción para Retenelo.

Dos responsabilidades:
  1. ``train()``   — Entrena XGBoost sobre el dataset sintético y persiste el modelo
     y los encoders en ``config.MODELS_DIR``.
  2. ``predict()`` — Dado un evento de pago fallido, evalúa la grilla de acciones
     (140 combinaciones) y devuelve la triple óptima (ventana, canal, tono) con la
     mayor P(recovered).

Arquitectura:
  - XGBoostClassifier: target = ``recovered`` (binario). Las acciones se incluyen como
    features → el modelo aprende qué acción funciona en qué contexto.
  - Selección de acción: se evalúan las 7 ventanas × 5 canales × 4 tonos; se retiene
    la combinación con mayor P(recovered).
  - Epsilon-greedy: para clientes nuevos (prior_recoveries == 0, tenure_meses <= 3)
    se aplica exploración aleatoria con prob. EPSILON para fomentar diversidad.
  - Fallback rule-based: si el modelo aún no está entrenado, la recomendación sale
    directamente de los catálogos JSON (decline_codes + archetypes).

Compliance (no opcional):
  - Si ``attempt_number >= PAYWAY_MAX_INTENTOS``: compliance_ok = False, no reintentar.
  - Si el código tiene ``never_retry = True``: mismo resultado.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from data import salary_calendar as sc

# ---------------------------------------------------------------------------
# Feature schema (must stay in sync with generator.py COLUMNAS)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "monto_ars", "attempt_number", "tokenized", "card_expiry_delta_dias",
    "tenure_meses", "prior_recoveries", "engagement_recency_dias", "arpu_ars",
    "day_of_month", "days_to_quincena", "days_to_fin_de_mes",
    "is_aguinaldo_month", "anses_pay_flag",
    "indec_ipc_mom", "bcra_rate", "billetera_yield_proxy",
]

CAT_FEATURES: list[str] = [
    "decline_code", "decline_tipo", "red", "acquirer", "card_type",
    "employment_type", "digital_literacy", "archetype_id",
]

ACTION_FEATURES: list[str] = ["action_window", "action_channel", "action_tone"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CAT_FEATURES + ACTION_FEATURES

TARGET = "recovered"

# Epsilon-greedy exploration rate for new clients.
EPSILON = 0.10

# Numeric defaults for missing event fields (median-like estimates for Argentina).
_NUMERIC_DEFAULTS: dict[str, float] = {
    "monto_ars": 7000.0,
    "tokenized": 1.0,
    "card_expiry_delta_dias": 300.0,
    "tenure_meses": 12.0,
    "prior_recoveries": 0.0,
    "engagement_recency_dias": 15.0,
    "arpu_ars": 7000.0,
    "indec_ipc_mom": 0.045,
    "bcra_rate": 0.55,
    "billetera_yield_proxy": 0.42,
}

# Categorical defaults (most common values in Argentina).
_CAT_DEFAULTS: dict[str, str] = {
    "red": "Visa",
    "acquirer": "Payway",
    "card_type": "credito",
}


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------

@dataclass
class RecoveryRecommendation:
    """Triple de acción óptima + metadata de compliance."""
    ventana: str
    canal: str
    tono: str
    recovery_prob: float
    attempt_number: int
    compliance_ok: bool
    decline_code: str
    archetype_id: str
    source: str = "model"  # "model" | "fallback_rules"
    compliance_reason: str = ""


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _load_catalogs() -> tuple[dict, dict]:
    """Carga y cachea arquetipos y códigos de rechazo."""
    with open(config.ARCHETYPES_PATH, encoding="utf-8") as f:
        archetypes = json.load(f)["archetypes"]
    with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
        codes = json.load(f)["codes"]
    return archetypes, codes


_CATALOGS: tuple[dict, dict] | None = None


def get_catalogs() -> tuple[dict, dict]:
    global _CATALOGS
    if _CATALOGS is None:
        _CATALOGS = _load_catalogs()
    return _CATALOGS


# ---------------------------------------------------------------------------
# Compliance gate (non-optional)
# ---------------------------------------------------------------------------

def _compliance_check(attempt_number: int, decline_code: str) -> tuple[bool, str]:
    """Devuelve (ok, motivo). Gate duro: nunca reintentar si aplica."""
    if attempt_number >= config.PAYWAY_MAX_INTENTOS:
        return False, f"Límite Payway alcanzado ({attempt_number}/{config.PAYWAY_MAX_INTENTOS} intentos)"

    _, codes = get_catalogs()
    code_info = codes.get(str(decline_code), {})
    if code_info.get("never_retry", False):
        return False, f"Código {decline_code} nunca se reintenta (tarjeta robada/perdida)"

    return True, ""


# ---------------------------------------------------------------------------
# Rule-based fallback (no model needed)
# ---------------------------------------------------------------------------

def _fallback_recommendation(event: dict) -> RecoveryRecommendation:
    """Recomendación basada en reglas cuando el modelo no está entrenado."""
    archetypes, codes = get_catalogs()
    decline_code = str(event.get("decline_code", "05"))
    archetype_id = str(event.get("archetype_id", "empleado_privado_formal"))
    attempt_number = int(event.get("attempt_number", 1))

    compliance_ok, compliance_reason = _compliance_check(attempt_number, decline_code)

    code_info = codes.get(decline_code, codes.get("05"))
    arch = archetypes.get(archetype_id, archetypes.get("empleado_privado_formal"))

    if code_info.get("requiere_ventana_salarial", False):
        ventana = arch.get("ventana_salarial", "fin_de_mes")
    else:
        ventana = code_info.get("ventana_sugerida") or "inmediata"

    canal = arch.get("best_channel", code_info.get("canal_sugerido", "whatsapp"))
    tono = arch.get("tono_preferido", code_info.get("tono_sugerido", "cercano"))

    return RecoveryRecommendation(
        ventana=ventana,
        canal=canal,
        tono=tono,
        recovery_prob=arch.get("base_recovery_rate", 0.40),
        attempt_number=attempt_number,
        compliance_ok=compliance_ok,
        decline_code=decline_code,
        archetype_id=archetype_id,
        source="fallback_rules",
        compliance_reason=compliance_reason,
    )


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _model_exists() -> bool:
    return config.MODEL_PATH.exists() and config.ENCODERS_PATH.exists()


def _save_model(clf, encoders: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(config.MODEL_PATH))
    with open(config.ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)


def _load_model_and_encoders():
    """Carga el modelo y los encoders serializados. Raises FileNotFoundError si faltan."""
    if not _model_exists():
        raise FileNotFoundError(
            "Modelo no entrenado. Ejecutá: python model.py --train"
        )
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(str(config.MODEL_PATH))
    with open(config.ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return clf, encoders


# Lazy-loaded singleton for the Streamlit session.
_MODEL_CACHE: tuple | None = None


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_model_and_encoders()
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _enrich_event(event: dict) -> dict:
    """Completa campos faltantes del evento con defaults razonables."""
    e = dict(event)

    fecha_raw = e.get("fecha_evento")
    if isinstance(fecha_raw, date):
        fecha = fecha_raw
    elif isinstance(fecha_raw, str):
        fecha = date.fromisoformat(fecha_raw)
    else:
        fecha = date.today()

    temporal = sc.features_temporales(fecha)
    for k, v in temporal.items():
        e.setdefault(k, v)

    for col, default in _NUMERIC_DEFAULTS.items():
        e.setdefault(col, default)

    for col, default in _CAT_DEFAULTS.items():
        e.setdefault(col, default)

    archetypes, codes = get_catalogs()
    arch_id = str(e.get("archetype_id", "empleado_privado_formal"))
    arch = archetypes.get(arch_id, archetypes.get("empleado_privado_formal"))

    code_str = str(e.get("decline_code", "05"))
    code_info = codes.get(code_str, codes.get("05"))

    e.setdefault("decline_tipo", code_info.get("tipo", "soft"))
    e.setdefault("employment_type", arch.get("employment_type", "empleado_privado"))
    e.setdefault("digital_literacy", arch.get("digital_literacy", "media"))

    return e


def _encode_features(rows: list[dict], encoders: dict) -> np.ndarray:
    """Convierte una lista de dicts a matriz numpy con encoding aplicado."""
    df = pd.DataFrame(rows)[ALL_FEATURES]

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CAT_FEATURES + ACTION_FEATURES:
        le = encoders[col]
        df[col] = df[col].astype(str)
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])

    return df.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(dataset_path: Optional[Path] = None) -> None:
    """Entrena el modelo XGBoost y persiste model + encoders.

    Usa el CSV generado por generator.py. Requiere que el dataset exista.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    from xgboost import XGBClassifier

    path = Path(dataset_path) if dataset_path else config.DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}. Ejecutá: python generator.py"
        )

    print(f"[model] Cargando dataset: {path}")
    df = pd.read_csv(path)
    print(f"[model] {len(df)} filas — recovered media: {df[TARGET].mean():.3f}")

    # Drop non-feature columns.
    drop_cols = {"id_evento", "fecha_evento", "user_id", "bin", "contexto_cliente",
                 "recovery_day_offset", TARGET}
    feature_cols = [c for c in ALL_FEATURES if c not in drop_cols and c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"[model] Advertencia: columnas faltantes en el dataset: {missing}")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    encoders: dict = {}
    for col in CAT_FEATURES + ACTION_FEATURES:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoders[col] = le

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    y_prob = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    print(f"\n[model] AUC validación: {auc:.4f}")
    print(classification_report(y_val, (y_prob >= 0.5).astype(int), digits=3))

    _save_model(clf, encoders)
    print(f"[model] Modelo guardado en {config.MODEL_PATH}")
    print(f"[model] Encoders guardados en {config.ENCODERS_PATH}")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(event: dict, *, explore: bool | None = None) -> RecoveryRecommendation:
    """Devuelve la triple óptima (ventana, canal, tono) para el evento dado.

    Si el modelo no está entrenado, usa el fallback rule-based.
    El flag ``explore`` fuerza o deshabilita la exploración epsilon-greedy;
    si es None, se auto-detecta por perfil del cliente.
    """
    decline_code = str(event.get("decline_code", "05"))
    archetype_id = str(event.get("archetype_id", "empleado_privado_formal"))
    attempt_number = int(event.get("attempt_number", 1))

    compliance_ok, compliance_reason = _compliance_check(attempt_number, decline_code)

    if not _model_exists():
        rec = _fallback_recommendation(event)
        rec.compliance_ok = compliance_ok
        rec.compliance_reason = compliance_reason
        return rec

    try:
        clf, encoders = _get_model()
    except Exception as e:
        print(f"[model] Error cargando modelo ({e}); usando fallback rules.")
        rec = _fallback_recommendation(event)
        rec.compliance_ok = compliance_ok
        rec.compliance_reason = compliance_reason
        return rec

    base = _enrich_event(event)

    # Epsilon-greedy: explore if new client.
    if explore is None:
        is_new_client = (
            int(base.get("prior_recoveries", 0)) == 0
            and int(base.get("tenure_meses", 99)) <= 3
        )
        explore = is_new_client and (np.random.random() < EPSILON)

    if explore:
        rng = np.random.default_rng()
        return RecoveryRecommendation(
            ventana=str(rng.choice(config.VENTANAS_REINTENTO)),
            canal=str(rng.choice(config.CANALES)),
            tono=str(rng.choice(config.TONOS)),
            recovery_prob=float(base.get("base_recovery_rate", 0.4)),
            attempt_number=attempt_number,
            compliance_ok=compliance_ok,
            decline_code=decline_code,
            archetype_id=archetype_id,
            source="model_explore",
            compliance_reason=compliance_reason,
        )

    # Build action grid: 7 ventanas x 5 canales x 4 tonos = 140 candidates.
    action_grid = list(product(config.VENTANAS_REINTENTO, config.CANALES, config.TONOS))

    rows = []
    for ventana, canal, tono in action_grid:
        row = dict(base)
        row["action_window"] = ventana
        row["action_channel"] = canal
        row["action_tone"] = tono
        rows.append(row)

    try:
        X = _encode_features(rows, encoders)
        probs = clf.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"[model] Error en predicción ({e}); usando fallback rules.")
        rec = _fallback_recommendation(event)
        rec.compliance_ok = compliance_ok
        rec.compliance_reason = compliance_reason
        return rec

    best_idx = int(np.argmax(probs))
    best_ventana, best_canal, best_tono = action_grid[best_idx]

    return RecoveryRecommendation(
        ventana=best_ventana,
        canal=best_canal,
        tono=best_tono,
        recovery_prob=float(probs[best_idx]),
        attempt_number=attempt_number,
        compliance_ok=compliance_ok,
        decline_code=decline_code,
        archetype_id=archetype_id,
        source="model",
        compliance_reason=compliance_reason,
    )


# ---------------------------------------------------------------------------
# Public helper for app.py
# ---------------------------------------------------------------------------

def build_event(
    decline_code: str,
    archetype_id: str,
    attempt_number: int = 1,
    fecha: Optional[date] = None,
    **kwargs,
) -> dict:
    """Construye un dict de evento mínimo para pasar a ``predict()``.

    Los campos opcionales (monto, tenure, etc.) se pueden pasar como kwargs.
    """
    base: dict = {
        "decline_code": decline_code,
        "archetype_id": archetype_id,
        "attempt_number": attempt_number,
        "fecha_evento": (fecha or date.today()).isoformat(),
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo de recuperación Retenelo.")
    parser.add_argument(
        "--train", action="store_true",
        help="Entrena el modelo sobre el dataset sintético.",
    )
    parser.add_argument(
        "--dataset", type=str, default=str(config.DATASET_PATH),
        help="Ruta al CSV de entrenamiento.",
    )
    parser.add_argument(
        "--predict-demo", action="store_true",
        help="Corre una predicción de demo con un evento sintético.",
    )
    args = parser.parse_args()

    if args.train:
        train(dataset_path=args.dataset)

    if args.predict_demo:
        event = build_event(
            decline_code="51",
            archetype_id="empleado_publico",
            attempt_number=2,
        )
        rec = predict(event)
        print("\n--- Demo de predicción ---")
        print(f"  Decline: {rec.decline_code} | Arquetipo: {rec.archetype_id}")
        print(f"  Ventana: {rec.ventana} | Canal: {rec.canal} | Tono: {rec.tono}")
        print(f"  P(recovered): {rec.recovery_prob:.3f}")
        print(f"  Compliance OK: {rec.compliance_ok}")
        if rec.compliance_reason:
            print(f"  Motivo: {rec.compliance_reason}")
        print(f"  Fuente: {rec.source}")


if __name__ == "__main__":
    main()
