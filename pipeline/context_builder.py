# === context_builder.py ===
"""Enriquece el perfil de un usuario con tres capas de contexto.

Capas:
  1. service_ctx  — comportamiento lector + señales de engagement (Vertical: Biblioteca)
  2. economic_ctx — snapshot macro argentino: inflación, calendario salarial, ventana aguinaldo
  3. geo_ctx      — tier geográfico de la provincia/ciudad del usuario

El dict resultante se pasa tal cual a ``output_composer.compose()`` y a
``recovery_engine.get_action_triple()``. No contiene datos sensibles (cumple Ley 25.326).

Uso:
    from context_builder import build_context
    ctx = build_context(user_dict, decline_code="51", reference_date=date.today())
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import config
from data.salary_calendar import (
    dias_hasta_quincena,
    dias_hasta_fin_de_mes,
    es_mes_aguinaldo,
    ventanas_aguinaldo,
    proxima_ventana_cobro,
)


# ---------------------------------------------------------------------------
# Loaders (module-level cache)
# ---------------------------------------------------------------------------

_GEO_TIERS: dict | None = None
_DECLINE_CODES: dict | None = None


def _get_geo_tiers() -> dict:
    global _GEO_TIERS
    if _GEO_TIERS is None:
        with open(config.GEO_TIERS_PATH, encoding="utf-8") as f:
            _GEO_TIERS = json.load(f)
    return _GEO_TIERS


def _get_decline_codes() -> dict:
    global _DECLINE_CODES
    if _DECLINE_CODES is None:
        with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
            _DECLINE_CODES = json.load(f)["codes"]
    return _DECLINE_CODES


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def _build_service_ctx(user_dict: dict) -> dict[str, Any]:
    """Extrae señales de comportamiento lector desde el perfil del usuario."""
    freq_raw = str(user_dict.get("frecuencia_apertura_app", "monthly")).lower()
    freq_map = {"daily": "alta", "weekly": "media", "monthly": "baja"}

    return {
        "libros_leidos_total": int(user_dict.get("libros_leidos_total", 0)),
        "libros_leidos_ultimos_3_meses": int(user_dict.get("libros_leidos_ultimos_3_meses", 0)),
        "rating_promedio": float(user_dict.get("rating_promedio_dado", 0.0)),
        "autor_favorito": user_dict.get("autor_favorito", None),
        "genero_favorito": user_dict.get("genero_favorito", None),
        "frecuencia_uso": freq_map.get(freq_raw, "baja"),
        "ultimo_acceso_dias": int(user_dict.get("ultimo_acceso_dias", 30)),
        "lista_deseos_activa": bool(user_dict.get("lista_deseos_activa", False)),
        "resenas_escritas": int(user_dict.get("resenas_escritas", 0)),
        "engagement_nivel": _engagement_level(user_dict),
    }


def _engagement_level(user_dict: dict) -> str:
    """Clasifica el nivel de engagement en alto / medio / bajo."""
    libros = int(user_dict.get("libros_leidos_ultimos_3_meses", 0))
    acceso = int(user_dict.get("ultimo_acceso_dias", 30))
    freq = str(user_dict.get("frecuencia_apertura_app", "monthly")).lower()

    if libros >= 3 and acceso <= 7 and freq == "daily":
        return "alto"
    if libros >= 1 and acceso <= 30:
        return "medio"
    return "bajo"


def _build_economic_ctx(reference_date: date, dni_last_digit: int | None = None) -> dict[str, Any]:
    """Ensambla el snapshot económico argentino relevante para la fecha dada."""
    macro = config.MACRO_SNAPSHOT

    dias_q = dias_hasta_quincena(reference_date)
    dias_fdm = dias_hasta_fin_de_mes(reference_date)
    aguinaldo = es_mes_aguinaldo(reference_date)

    proxima_quincena = proxima_ventana_cobro(reference_date, "quincena")
    proxima_fdm = proxima_ventana_cobro(reference_date, "fin_de_mes")

    # Determine the nearest payday window label.
    if dias_q <= 3:
        ventana_proxima = "quincena_inminente"
    elif dias_fdm <= 3:
        ventana_proxima = "fin_de_mes_inminente"
    elif dias_q <= dias_fdm:
        ventana_proxima = "quincena"
    else:
        ventana_proxima = "fin_de_mes"

    # Is it an aguinaldo spike window?
    in_aguinaldo_window = False
    for inicio, fin in ventanas_aguinaldo(reference_date.year):
        if inicio <= reference_date <= fin:
            in_aguinaldo_window = True
            break

    return {
        "inflacion_mensual_pct": macro["inflacion_mensual_pct"],
        "inflacion_tier": macro["inflacion_tier"],
        "bcra_tna_pct": macro["bcra_tna_pct"],
        "billetera_yield_proxy_pct": macro["billetera_yield_proxy_pct"],
        "dias_hasta_quincena": dias_q,
        "dias_hasta_fin_de_mes": dias_fdm,
        "proxima_quincena": proxima_quincena.isoformat(),
        "proxima_fin_de_mes": proxima_fdm.isoformat(),
        "ventana_proxima": ventana_proxima,
        "es_mes_aguinaldo": aguinaldo,
        "en_ventana_aguinaldo": in_aguinaldo_window,
    }


def _build_geo_ctx(user_dict: dict) -> dict[str, Any]:
    """Determina el tier geográfico del usuario según provincia o ciudad."""
    geo = _get_geo_tiers()
    tier_defs = geo.get("tier_definitions", {})
    provinces = geo.get("provinces", {})
    cities = geo.get("cities", {})
    default_tier = int(geo.get("default_tier", 3))

    province = str(user_dict.get("state", user_dict.get("provincia", ""))).strip()
    city = str(user_dict.get("ciudad", user_dict.get("city", ""))).strip()

    tier = cities.get(city, provinces.get(province, default_tier))
    tier = int(tier)

    return {
        "provincia": province or None,
        "ciudad": city or None,
        "geo_tier": tier,
        "geo_tier_label": tier_defs.get(str(tier), "Interior"),
    }


def _build_decline_info(decline_code: str) -> dict[str, Any]:
    """Adjunta información legible del código de rechazo (sin datos internos de scoring)."""
    codes = _get_decline_codes()
    code_str = str(decline_code)
    info = codes.get(code_str, {})

    return {
        "codigo": code_str,
        "descripcion": info.get("descripcion", "Rechazo del emisor"),
        "tipo": info.get("tipo", "soft"),
        "recuperable": bool(info.get("recuperable", True)),
        "requiere_accion_usuario": info.get("auto_or_user", "auto") == "user",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context(
    user_dict: dict,
    decline_code: str,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Ensambla el contexto enriquecido de tres capas para un evento de pago fallido.

    Args:
        user_dict: Dict con datos del usuario (signup form + in-app behavior).
        decline_code: Código ISO-8583 del rechazo (ej. "51").
        reference_date: Fecha del evento. Defaults a hoy.

    Returns:
        Dict con claves service_ctx, economic_ctx, geo_ctx, decline_info.
        No contiene datos sensibles (Ley 25.326 compliant).
    """
    ref = reference_date or date.today()

    # Extract DNI last digit for ANSES calendar if available.
    cuit_dni = str(user_dict.get("cuit_dni", ""))
    dni_last = int(cuit_dni[-1]) if cuit_dni and cuit_dni[-1].isdigit() else None

    return {
        "service_ctx": _build_service_ctx(user_dict),
        "economic_ctx": _build_economic_ctx(ref, dni_last_digit=dni_last),
        "geo_ctx": _build_geo_ctx(user_dict),
        "decline_info": _build_decline_info(decline_code),
        "first_name": user_dict.get("first_name", ""),  # used for personalization only
    }
