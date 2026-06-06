"""Generador de dataset sintético para Retenelo.

División de trabajo:
  - numpy = backbone: TODA columna con distribución controlada y la etiqueta se
    muestrea con ``rng.choice(p=weights)`` y sorteos numéricos. Es lo que entrena el
    modelo: determinista (con seed), rápido, gratis y con distribución exacta.
  - Haiku = textura (opcional): solo la columna libre ``contexto_cliente``. Nunca toca
    la etiqueta ni los pesos. Si no hay API key o se pasa --no-haiku, se usa una
    plantilla numpy y el dataset sigue siendo válido.

Las distribuciones de arquetipos y códigos NO son uniformes: salen de
``prevalencia_weight`` en los catálogos JSON (config over hardcoding). El código 51
domina y los arquetipos raros quedan en minoría, reflejando la realidad argentina.

Anti-circularidad: la etiqueta ``recovered`` no es función determinista de las
features. Se calcula una probabilidad p (base del arquetipo x multiplicadores de
código/timing/canal/intentos) con ruido, y luego ``recovered ~ Bernoulli(p)``. Así el
modelo aprende una señal real y ruidosa (métricas creíbles, no perfectas).

Uso:
    python generator.py                 # backbone numpy (sin Haiku)
    python generator.py --rows 20000 --seed 42
    python generator.py --haiku         # agrega textura con claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data import salary_calendar as sc

# --- Parámetros de realismo sintético (estimaciones; no son reglas de negocio) ---
REDES = ["Visa", "Mastercard", "Cabal", "Amex", "Naranja"]
RED_WEIGHTS = [0.48, 0.30, 0.10, 0.06, 0.06]
RED_BIN_PREFIX = {"Visa": "4", "Mastercard": "5", "Cabal": "6", "Amex": "3", "Naranja": "5"}
RED_DEBITO_PROB = {"Cabal": 0.85, "Visa": 0.30, "Mastercard": 0.30, "Amex": 0.10, "Naranja": 0.20}

ACQUIRERS = ["Payway", "Fiserv", "Geopagos", "Other"]
ACQUIRER_WEIGHTS = [0.55, 0.20, 0.15, 0.10]

ARPU_BAND_MEAN_ARS = {"baja": 3500, "media": 7000, "media_alta": 12000, "alta": 22000}
TENURE_BAND_RANGO = {"baja": (1, 6), "media": (6, 24), "alta": (24, 72)}

# Distribución de intentos: Retenelo entra con attempt_number >= 1; cola hasta 15.
_ATTEMPTS = np.arange(1, 16)
_ATTEMPT_WEIGHTS = 0.7 ** (_ATTEMPTS - 1)
_ATTEMPT_WEIGHTS = _ATTEMPT_WEIGHTS / _ATTEMPT_WEIGHTS.sum()

# Probabilidad de muestrear la acción "alineada" (vs exploración aleatoria) en el log.
P_ACCION_ALINEADA = 0.6

# Rango temporal de los eventos sintéticos (hasta la fecha actual del proyecto).
FECHA_FIN = date(2026, 6, 6)
FECHA_INICIO = date(2025, 7, 1)

COLUMNAS = [
    "id_evento", "fecha_evento",
    "decline_code", "decline_tipo", "red", "acquirer", "bin", "card_type",
    "monto_ars", "attempt_number", "tokenized", "card_expiry_delta_dias",
    "user_id", "archetype_id", "employment_type", "digital_literacy",
    "tenure_meses", "prior_recoveries", "engagement_recency_dias", "arpu_ars",
    "day_of_month", "days_to_quincena", "days_to_fin_de_mes",
    "is_aguinaldo_month", "anses_pay_flag",
    "indec_ipc_mom", "bcra_rate", "billetera_yield_proxy",
    "action_window", "action_channel", "action_tone",
    "recovered", "recovery_day_offset",
    "contexto_cliente",
]


def _cargar_catalogos() -> tuple[dict, dict]:
    """Carga arquetipos y códigos de rechazo desde los JSON."""
    with open(config.ARCHETYPES_PATH, encoding="utf-8") as f:
        archetypes = json.load(f)["archetypes"]
    with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
        codes = json.load(f)["codes"]
    return archetypes, codes


def _pesos_normalizados(d: dict) -> tuple[list, np.ndarray]:
    """Devuelve (claves, pesos normalizados) a partir de un dict {clave: peso}."""
    claves = list(d.keys())
    pesos = np.array([d[k] if not isinstance(d[k], dict) else d[k]["prevalencia_weight"]
                      for k in claves], dtype=float)
    return claves, pesos / pesos.sum()


def _ventana_alineada(arch: dict, code_info: dict) -> str:
    """Ventana de reintento 'correcta' para este arquetipo+código."""
    if code_info["requiere_ventana_salarial"]:
        return arch["ventana_salarial"]
    return code_info["ventana_sugerida"] or "inmediata"


def _muestrear_codigo(arch: dict, codes_glob: list, pesos_glob: np.ndarray,
                      rng: np.random.Generator) -> str:
    """Mezcla afinidad del arquetipo con la prevalencia global (asegura hard declines)."""
    if rng.random() < config.GENERATOR_ALPHA_ARQUETIPO:
        claves, pesos = _pesos_normalizados(arch["likely_decline_codes"])
        return str(rng.choice(claves, p=pesos))
    return str(rng.choice(codes_glob, p=pesos_glob))


def _muestrear_accion(arch: dict, code_info: dict, rng: np.random.Generator) -> dict:
    """Acción registrada en el log sintético: alineada con prob P_ACCION_ALINEADA, si no aleatoria."""
    alineada = _ventana_alineada(arch, code_info)
    window = alineada if rng.random() < P_ACCION_ALINEADA else str(rng.choice(config.VENTANAS_REINTENTO))
    channel = arch["best_channel"] if rng.random() < P_ACCION_ALINEADA else str(rng.choice(config.CANALES))
    tone = arch["tono_preferido"] if rng.random() < P_ACCION_ALINEADA else str(rng.choice(config.TONOS))
    return {"action_window": window, "action_channel": channel, "action_tone": tone}


def _prob_recuperacion(arch: dict, code_info: dict, accion: dict, attempt_number: int,
                       cabal_debito: bool, rng: np.random.Generator) -> float:
    """Probabilidad latente de recuperación (con ruido). Base del label Bernoulli."""
    p = arch["base_recovery_rate"]

    # Código: los hard declines casi no se recuperan vía reintento.
    if code_info["never_retry"]:
        p *= 0.15
    elif not code_info["recuperable"]:
        p *= 0.30
    if cabal_debito:
        p *= 0.10  # débito automático Cabal ~ no colectable por reintento

    # Timing: la ventana alineada es la palanca de mayor impacto.
    alineada = _ventana_alineada(arch, code_info)
    if accion["action_window"] == alineada:
        p *= 1.25
    elif accion["action_window"] == "inmediata":
        p *= 0.85  # reintentar ya mismo un problema de fondos rinde poco
    else:
        p *= 0.90

    # Canal: respuesta del arquetipo al canal elegido (0..1 -> multiplicador 0.6..1.1).
    resp = arch["channel_responsiveness"].get(accion["action_channel"], 0.5)
    p *= 0.6 + 0.5 * resp

    # Tono: leve aporte si coincide con el preferido.
    if accion["action_tone"] == arch["tono_preferido"]:
        p *= 1.05

    # Intentos: más intentos -> más difícil (fatiga / caso genuinamente duro).
    p *= max(0.2, 1.0 - 0.05 * (attempt_number - 1))

    # Ruido gaussiano para evitar que el modelo memorice la fórmula.
    p *= rng.normal(1.0, 0.08)
    return float(np.clip(p, 0.01, 0.97))


def _columna_contexto(df: pd.DataFrame, archetypes: dict, usar_haiku: bool) -> pd.Series:
    """Textura libre por fila. Haiku si está disponible; si no, plantilla numpy."""
    notas = None
    if usar_haiku:
        try:
            notas = _haiku_notas_por_arquetipo(list(archetypes.values()))
        except Exception as e:  # ImportError, RuntimeError (sin key), etc.
            print(f"[generator] Haiku no disponible ({e}); uso plantillas numpy.")
    if notas:
        return df["archetype_id"].map(lambda a: notas.get(a, archetypes[a]["nombre"]))
    return df["archetype_id"].map(
        lambda a: f"{archetypes[a]['nombre']} ({archetypes[a]['financial_behavior']})"
    )


def _haiku_notas_por_arquetipo(archetypes_list: list[dict]) -> dict[str, str]:
    """Una sola llamada a Haiku que devuelve {archetype_id: nota corta}. [texture only].

    Se importa llm_client de forma perezosa: si todavía no existe o no hay API key,
    el caller cae a plantillas numpy.
    """
    from llm_client import completar_haiku_json  # lazy import (módulo posterior)

    perfiles = [{"id": a["id"], "nombre": a["nombre"],
                 "comportamiento": a["financial_behavior"]} for a in archetypes_list]
    prompt = (
        "Sos un generador de datos. Para cada arquetipo de cliente argentino, escribí una "
        "nota interna muy breve (max 12 palabras, en español rioplatense) que describa su "
        "contexto de pago. Devolvé SOLO un JSON {id: nota}.\n\n"
        f"Arquetipos: {json.dumps(perfiles, ensure_ascii=False)}"
    )
    return completar_haiku_json(prompt)


def build_dataset(n_rows: int, seed: int, usar_haiku: bool = False) -> pd.DataFrame:
    """Genera el DataFrame sintético completo (backbone numpy)."""
    rng = np.random.default_rng(seed)
    archetypes, codes = _cargar_catalogos()

    arch_ids, arch_w = _pesos_normalizados({k: v for k, v in archetypes.items()})
    code_ids, code_w = _pesos_normalizados({k: v for k, v in codes.items()})
    span = (FECHA_FIN - FECHA_INICIO).days

    filas = []
    for i in range(n_rows):
        archetype_id = str(rng.choice(arch_ids, p=arch_w))
        arch = archetypes[archetype_id]
        decline_code = _muestrear_codigo(arch, code_ids, code_w, rng)
        code_info = codes[decline_code]

        fecha = FECHA_INICIO + timedelta(days=int(rng.integers(0, span + 1)))
        feats = sc.features_temporales(fecha)

        red = str(rng.choice(REDES, p=RED_WEIGHTS))
        card_type = "debito" if rng.random() < RED_DEBITO_PROB[red] else "credito"
        cabal_debito = (red == "Cabal" and card_type == "debito")
        bin_num = RED_BIN_PREFIX[red] + "".join(str(rng.integers(0, 10)) for _ in range(5))
        acquirer = str(rng.choice(ACQUIRERS, p=ACQUIRER_WEIGHTS))

        arpu = float(rng.lognormal(np.log(ARPU_BAND_MEAN_ARS[arch["arpu_band"]]), 0.30))
        monto = round(arpu * rng.uniform(0.9, 1.1), -2)
        attempt_number = int(rng.choice(_ATTEMPTS, p=_ATTEMPT_WEIGHTS))
        tokenized = bool(rng.random() < 0.6)
        if decline_code == "54":
            card_expiry_delta = int(rng.integers(-365, -1))  # vencida
        else:
            card_expiry_delta = int(np.clip(rng.normal(300, 200), -60, 1000))

        tmin, tmax = TENURE_BAND_RANGO[arch["tenure_band"]]
        tenure = int(rng.integers(tmin, tmax + 1))
        prior_recoveries = int(rng.poisson(tenure / 18.0))
        engagement_recency = int(np.clip(rng.exponential(20), 0, 365))

        indec_ipc = float(np.clip(rng.normal(0.045, 0.015), 0.005, 0.15))
        bcra = float(np.clip(rng.normal(0.55, 0.08), 0.20, 1.00))
        billetera = float(bcra * rng.uniform(0.7, 0.95))

        accion = _muestrear_accion(arch, code_info, rng)
        p = _prob_recuperacion(arch, code_info, accion, attempt_number, cabal_debito, rng)
        recovered = int(rng.random() < p)
        if recovered:
            retry_date = sc.proxima_ventana_cobro(fecha, accion["action_window"])
            recovery_day_offset = max(0, (retry_date - fecha).days + int(rng.integers(0, 3)))
        else:
            recovery_day_offset = -1

        filas.append({
            "id_evento": f"EVT{i:07d}",
            "fecha_evento": fecha.isoformat(),
            "decline_code": decline_code,
            "decline_tipo": code_info["tipo"],
            "red": red,
            "acquirer": acquirer,
            "bin": bin_num,
            "card_type": card_type,
            "monto_ars": monto,
            "attempt_number": attempt_number,
            "tokenized": tokenized,
            "card_expiry_delta_dias": card_expiry_delta,
            "user_id": f"USR{int(rng.integers(0, n_rows * 3)):07d}",
            "archetype_id": archetype_id,
            "employment_type": arch["employment_type"],
            "digital_literacy": arch["digital_literacy"],
            "tenure_meses": tenure,
            "prior_recoveries": prior_recoveries,
            "engagement_recency_dias": engagement_recency,
            "arpu_ars": round(arpu, -2),
            "day_of_month": feats["day_of_month"],
            "days_to_quincena": feats["days_to_quincena"],
            "days_to_fin_de_mes": feats["days_to_fin_de_mes"],
            "is_aguinaldo_month": feats["is_aguinaldo_month"],
            "anses_pay_flag": feats["anses_pay_flag"],
            "indec_ipc_mom": round(indec_ipc, 4),
            "bcra_rate": round(bcra, 4),
            "billetera_yield_proxy": round(billetera, 4),
            "action_window": accion["action_window"],
            "action_channel": accion["action_channel"],
            "action_tone": accion["action_tone"],
            "recovered": recovered,
            "recovery_day_offset": recovery_day_offset,
        })

    df = pd.DataFrame(filas)
    df["contexto_cliente"] = _columna_contexto(df, archetypes, usar_haiku)
    return df[COLUMNAS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de dataset sintético de Retenelo.")
    parser.add_argument("--rows", type=int, default=config.GENERATOR_DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=config.GENERATOR_DEFAULT_SEED)
    parser.add_argument("--haiku", action="store_true", help="Agrega textura con claude-haiku-4-5.")
    parser.add_argument("--out", type=str, default=str(config.DATASET_PATH))
    args = parser.parse_args()

    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args.rows, args.seed, usar_haiku=args.haiku)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"OK - {len(df)} filas -> {args.out}")
    print(f"  recovered media: {df['recovered'].mean():.3f}")
    print("  top arquetipos:", df["archetype_id"].value_counts(normalize=True).head(4).round(3).to_dict())
    print("  top códigos:", df["decline_code"].value_counts(normalize=True).head(4).round(3).to_dict())


if __name__ == "__main__":
    main()
