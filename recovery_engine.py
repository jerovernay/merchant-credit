# === recovery_engine.py ===
"""Motor de recuperación: traduce el contexto enriquecido en un action triple concreto.

Este módulo es un orquestador delgado. La inteligencia de scoring vive en ``model.py``;
las reglas de dominio viven en ``data/decline_codes.json``. Este módulo:
  1. Construye el evento que espera model.predict().
  2. Llama al modelo (o al fallback rule-based si no está entrenado).
  3. Traduce la ventana string en un datetime concreto.
  4. Aplica el override de Cabal débito (leído del JSON).
  5. Enforcea el tope de 15 intentos — devuelve channel="none" si se alcanzó.

Uso:
    from recovery_engine import get_action_triple
    triple = get_action_triple(
        user_type="lector_voraz",
        enriched_context=ctx,
        decline_code="51",
        attempt_number=2,
    )
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any

import config
from data.salary_calendar import proxima_ventana_cobro
import model as ml


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

_DECLINE_DATA: dict | None = None


def _get_decline_data() -> dict:
    global _DECLINE_DATA
    if _DECLINE_DATA is None:
        with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
            _DECLINE_DATA = json.load(f)
    return _DECLINE_DATA


def _get_cabal_override() -> dict:
    """Lee el override de Cabal débito desde decline_codes.json."""
    return _get_decline_data().get("overrides", {}).get("cabal_debito", {})


# ---------------------------------------------------------------------------
# Window resolver
# ---------------------------------------------------------------------------

def _resolve_window(ventana_key: str, reference_date: date) -> datetime | None:
    """Convierte la clave de ventana (string) en un datetime concreto.

    Devuelve None solo si la acción es "none" (no reintentar).
    """
    if ventana_key == "none" or not ventana_key:
        return None

    target_date = proxima_ventana_cobro(reference_date, ventana_key)
    # Schedule at 10:00 local (good open-rate window for Argentina).
    return datetime(target_date.year, target_date.month, target_date.day, 10, 0, 0)


# ---------------------------------------------------------------------------
# Cabal débito override
# ---------------------------------------------------------------------------

def _is_cabal_debito(enriched_context: dict) -> bool:
    """True si el medio de pago es Cabal débito (requiere acción del usuario)."""
    svc = enriched_context.get("service_ctx", {})
    red = str(svc.get("red", "")).lower()
    card_type = str(svc.get("card_type", "")).lower()
    return red == "cabal" and card_type == "debito"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_action_triple(
    user_type: str,
    enriched_context: dict[str, Any],
    decline_code: str,
    attempt_number: int,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Devuelve la triple de acción óptima para el evento de pago fallido.

    Args:
        user_type: Etiqueta de tipo de usuario asignada por clustering.py.
        enriched_context: Dict enriquecido construido por context_builder.build_context().
        decline_code: Código ISO-8583 del rechazo (ej. "51").
        attempt_number: Número de intento actual (siempre >= 1).
        reference_date: Fecha base del evento. Defaults a hoy.

    Returns:
        {
            "retry_window": datetime | None,
            "channel": str,              # whatsapp|email|push|sms|llamada|none
            "tone": str,
            "compliance_ok": bool,
            "reason": str,
            "recovery_prob": float,
            "source": str,
        }
    """
    ref = reference_date or date.today()

    # --- Hard compliance gate (before any ML call) ---
    if attempt_number >= config.PAYWAY_MAX_INTENTOS:
        return {
            "retry_window": None,
            "channel": "none",
            "tone": "informativo",
            "compliance_ok": False,
            "reason": f"Límite Payway alcanzado ({attempt_number}/{config.PAYWAY_MAX_INTENTOS} intentos).",
            "recovery_prob": 0.0,
            "source": "compliance_gate",
        }

    # --- Cabal débito override (read from JSON, not hardcoded) ---
    if _is_cabal_debito(enriched_context):
        override = _get_cabal_override()
        return {
            "retry_window": None,
            "channel": override.get("canal_sugerido", "whatsapp"),
            "tone": override.get("tono_sugerido", "cercano"),
            "compliance_ok": True,
            "reason": override.get("notas", "Cabal débito: acción del usuario requerida."),
            "recovery_prob": 0.15,
            "source": "cabal_override",
        }

    # --- Build model event dict ---
    eco = enriched_context.get("economic_ctx", {})
    geo = enriched_context.get("geo_ctx", {})
    svc = enriched_context.get("service_ctx", {})

    event = ml.build_event(
        decline_code=decline_code,
        archetype_id=user_type,  # model accepts user_type as archetype_id proxy
        attempt_number=attempt_number,
        fecha=ref,
        # Pass macro context through to model's numeric features.
        indec_ipc_mom=eco.get("inflacion_mensual_pct", config.MACRO_SNAPSHOT["inflacion_mensual_pct"]) / 100,
        bcra_rate=eco.get("bcra_tna_pct", config.MACRO_SNAPSHOT["bcra_tna_pct"]) / 100,
        billetera_yield_proxy=eco.get("billetera_yield_proxy_pct", config.MACRO_SNAPSHOT["billetera_yield_proxy_pct"]) / 100,
        engagement_recency_dias=svc.get("ultimo_acceso_dias", 15),
    )

    rec = ml.predict(event)

    # --- Compliance check already embedded in model.predict ---
    if not rec.compliance_ok:
        return {
            "retry_window": None,
            "channel": "none",
            "tone": "informativo",
            "compliance_ok": False,
            "reason": rec.compliance_reason or "Código de rechazo no reintentable.",
            "recovery_prob": 0.0,
            "source": rec.source,
        }

    retry_dt = _resolve_window(rec.ventana, ref)

    return {
        "retry_window": retry_dt,
        "channel": rec.canal,
        "tone": rec.tono,
        "compliance_ok": True,
        "reason": "",
        "recovery_prob": round(rec.recovery_prob, 4),
        "source": rec.source,
    }
