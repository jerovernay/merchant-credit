"""Configuración central de Retenelo.

Todas las reglas de negocio (umbrales, vocabularios, rutas) viven acá para que los
módulos de lógica nunca tengan números mágicos. Términos de dominio en español;
comentarios internos en inglés.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Rutas (paths resolved relative to project root) ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"

ARCHETYPES_PATH = DATA_DIR / "archetypes.json"
DECLINE_CODES_PATH = DATA_DIR / "decline_codes.json"
DATASET_PATH = SYNTHETIC_DIR / "dataset.csv"
SCHEMA_PATH = SYNTHETIC_DIR / "schema.md"
MODEL_PATH = MODELS_DIR / "model.json"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"

# --- Payway / regulatorio (Ley 25.326, BCRA) ---
# Hard cap enforced at engine level (recovery_engine.py), never at the UI.
PAYWAY_MAX_INTENTOS = 15            # tope de reintentos por transacción rechazada
PAYWAY_VENTANA_DIAS = 30           # ventana móvil (días) sobre la que aplica el tope
MONTO_MINIMO_RECUPERABLE_ARS = 500  # por debajo no conviene reintentar [PENDING calibración]

# --- Vocabularios de acción (candidate-action grid) ---
CANALES = ["whatsapp", "email", "sms", "push", "llamada"]
TONOS = ["cercano", "formal", "empatico", "urgente_suave"]
VENTANAS_REINTENTO = [
    "inmediata",
    "quincena",
    "post_quincena",
    "fin_de_mes",
    "anses",
    "post_aguinaldo",
    "post_fecha_resumen",
]

# --- Proxy LLM (LiteLLM sobre AWS Bedrock, hackathon) ---
LLM_BASE_URL = "https://litellm-alb-1708856422.us-east-1.elb.amazonaws.com"
MODELO_HAIKU = "claude-haiku-4-5"     # data generation (texture only)
MODELO_SONNET = "claude-sonnet-4-6"   # final message composition only
LLM_VERIFY_TLS = False                # hackathon proxy uses a self-signed cert
LLM_API_KEY_ENV = "RETENELO_LLM_API_KEY"  # env var name that holds the proxy API key


def get_llm_api_key() -> str:
    """Lee la API key del proxy desde el entorno; error claro si falta."""
    key = os.environ.get(LLM_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Falta la API key del LLM. Definí la variable de entorno {LLM_API_KEY_ENV}."
        )
    return key


# --- Defaults del simulador de ROI (benchmark SaaS argentino mid-market) ---
ROI_DEFAULTS = {
    "subscriber_count": 1000,
    "arpu_ars": 5000,
    "monthly_failure_rate": 0.08,
    "recovery_rate": 0.40,
    "commission_pct": 0.125,
    "retention_months": 6,
}

# --- Defaults del generador de datos sintéticos ---
GENERATOR_DEFAULT_ROWS = 20000
GENERATOR_DEFAULT_SEED = 42
# Proporción en que el código de rechazo se muestrea desde la afinidad del arquetipo
# vs la prevalencia global. Garantiza que los hard declines globales también aparezcan.
GENERATOR_ALPHA_ARQUETIPO = 0.7

# --- Vertical 1: Biblioteca ---
BIBLIOTECA_PATH = DATA_DIR / "services" / "biblioteca.json"
GEO_TIERS_PATH = DATA_DIR / "geo_tiers.json"
KMEANS_N_CLUSTERS = 5  # configurable cluster count for clustering.py
KMEANS_BIBLIOTECA_PATH = MODELS_DIR / "kmeans_biblioteca.pkl"
KMEANS_SCALER_PATH = MODELS_DIR / "kmeans_scaler.pkl"

# --- Macro económico (snapshot — actualizar periódicamente) ---
MACRO_SNAPSHOT = {
    "inflacion_mensual_pct": 3.5,       # IPC MoM estimado (%)
    "bcra_tna_pct": 32.0,              # Tasa de política monetaria BCRA (%)
    "billetera_yield_proxy_pct": 28.0, # Yield promedio billeteras virtuales (%)
    "inflacion_tier": "moderada",       # baja | moderada | alta | hiperinflacion
}
