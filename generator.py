# === generator.py ===
"""Generador de dataset sintético para Vertical 1: Biblioteca de suscripción.

Genera eventos de pago fallido con todas las señales de signup e in-app behavior.
Distribuciones: código 51 ~50%, ciudades sesgadas AMBA ~60%, DNI uniform,
ultimo_acceso_dias sesgado a reciente.

Uso:
    python generator.py                    # 1000 rows, seed=42
    python generator.py --rows 5000 --seed 123
    python generator.py --out custom_path.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _load_biblioteca_config() -> dict:
    """Carga configuración de Vertical 1 desde data/services/biblioteca.json."""
    with open(config.BIBLIOTECA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_decline_codes() -> dict:
    """Carga códigos de rechazo desde decline_codes.json."""
    with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
        return json.load(f)["codes"]


def _generate_firstname(rng: np.random.Generator) -> str:
    """Nombres argentinos frecuentes."""
    nombres = [
        "Juan", "María", "Carlos", "Ana", "José", "Laura", "Roberto", "Patricia",
        "Diego", "Sandra", "Fernando", "Claudia", "Andrés", "Mónica", "Martín",
        "Lorena", "Miguel", "Elena", "Pablo", "Silvia", "Ricardo", "Cecilia",
        "Gustavo", "Roxana", "Sergio", "Verónica", "Raúl", "Nora", "Francisco",
        "Gabriela", "Lucas", "Alejandra", "Tomás", "Daniela", "Javier", "Valeria"
    ]
    return str(rng.choice(nombres))


def _generate_lastname(rng: np.random.Generator) -> str:
    """Apellidos argentinos frecuentes."""
    apellidos = [
        "García", "López", "González", "Martínez", "Rodríguez", "Pérez", "Sánchez",
        "Ramirez", "Flores", "Rivera", "Cruz", "Moreno", "Gutiérrez", "Ortiz",
        "Jiménez", "Hernández", "Vargas", "Castillo", "Rojas", "Díaz", "Santos",
        "Morales", "Reyes", "Domínguez", "Vega", "Salazar", "Campos", "Núñez",
        "Fuentes", "Medina", "Delgado", "Silva", "Carrillo", "Ruiz", "Espinoza"
    ]
    return str(rng.choice(apellidos))


def _generate_cuit_dni(rng: np.random.Generator) -> str:
    """DNI/CUIT argentino realista (11 dígitos para CUIT, 8 para DNI; usa CUIT)."""
    prefix = "23"  # CUIT prefix (simplificado; en realidad varía)
    dni = str(rng.integers(10000000, 99999999))
    verifier = str(rng.integers(0, 10))
    return prefix + dni + verifier


def _generate_ciudad_provincia(rng: np.random.Generator, bib_config: dict) -> tuple[str, str]:
    """Elige ciudad realista sesgada a AMBA, luego provincia correspondiente."""
    cities_dist = bib_config.get("city_distribution", {})

    tiers = ["amba", "provincial", "interior"]
    weights = [cities_dist.get(t, {}).get("weight", 0.33) for t in tiers]
    weights = np.array(weights) / np.array(weights).sum()

    tier = str(rng.choice(tiers, p=weights))
    cities_list = cities_dist.get(tier, {}).get("cities", ["Buenos Aires"])
    ciudad = str(rng.choice(cities_list))

    geo_tiers = _load_geo_tiers()
    provincia = geo_tiers.get("cities", {}).get(ciudad)
    if not provincia:
        provincia = geo_tiers.get("provinces", {}).get(ciudad, "Buenos Aires")

    provincia_name = _get_provincia_nombre(provincia, geo_tiers)

    return ciudad, provincia_name


def _load_geo_tiers() -> dict:
    """Carga geo_tiers.json para mapping ciudad -> provincia."""
    geo_path = config.DATA_DIR / "geo_tiers.json"
    with open(geo_path, encoding="utf-8") as f:
        return json.load(f)


def _get_provincia_nombre(tier_or_name: str | int, geo_tiers: dict) -> str:
    """Resuelve nombre de provincia desde tier o nombre."""
    if isinstance(tier_or_name, int):
        tier_map = {1: "Buenos Aires", 2: "Córdoba", 3: "Interior"}
        return tier_map.get(tier_or_name, "Buenos Aires")
    return str(tier_or_name)


def _generate_address(rng: np.random.Generator) -> tuple[str, str, str]:
    """street_address, postcode (simulado)."""
    streets = ["Avenida", "Calle", "Pasaje", "Boulevard", "Camino"]
    street_names = ["9 de Julio", "Corrientes", "Libertad", "Santa Fe", "Belgrano",
                   "Rivadavia", "Mitre", "Tucumán", "Sarmiento", "Lavalle"]
    number = str(rng.integers(100, 9999))
    street = f"{str(rng.choice(streets))} {str(rng.choice(street_names))} {number}"

    postcode = str(rng.integers(1000, 9999))

    return street, postcode


def _generate_phone(rng: np.random.Generator) -> str:
    """Teléfono argentino: +54 9 (area) (number)."""
    area = str(rng.integers(200, 399))
    number = str(rng.integers(10000000, 99999999))
    return f"+54 9 {area} {number}"


def _generate_email(first_name: str, last_name: str, rng: np.random.Generator) -> str:
    """Email realista."""
    domains = ["gmail.com", "yahoo.com.ar", "hotmail.com", "outlook.com", "yahoo.com"]
    separator = str(rng.choice([".", "_", ""]))
    domain = str(rng.choice(domains))
    return f"{first_name.lower()}{separator}{last_name.lower()}@{domain}"


def _generate_libros_leidos(rng: np.random.Generator) -> int:
    """Cantidad de libros leídos (sesgado a bajo)."""
    return int(np.clip(rng.exponential(8), 0, 100))


def _generate_libros_3m(libros_total: int, rng: np.random.Generator) -> int:
    """Libros leídos en últimos 3 meses (correlacionado con total)."""
    frac = rng.uniform(0, min(0.6, 1.0))
    return int(np.clip(libros_total * frac, 0, 15))


def _generate_rating_promedio(rng: np.random.Generator) -> float:
    """Rating promedio dado (0-5, sesgado a alto)."""
    return float(np.clip(rng.beta(5, 2) * 5, 0, 5))


def _generate_frecuencia_apertura(rng: np.random.Generator) -> str:
    """Frecuencia de apertura: daily, weekly, monthly."""
    opciones = ["daily", "weekly", "monthly"]
    pesos = np.array([0.20, 0.40, 0.40])
    return str(rng.choice(opciones, p=pesos))


def _generate_ultimo_acceso_dias(rng: np.random.Generator) -> int:
    """Días desde último acceso (sesgado a reciente)."""
    return int(np.clip(rng.exponential(15), 0, 365))


def _generate_lista_deseos(rng: np.random.Generator) -> bool:
    """Tiene lista de deseos activa."""
    return bool(rng.random() < 0.45)


def _generate_resenas_escritas(rng: np.random.Generator) -> int:
    """Cantidad de reseñas escritas (mayoría 0)."""
    if rng.random() < 0.7:
        return 0
    return int(np.clip(rng.poisson(3), 0, 50))


def _generate_autor_favorito(bib_config: dict, rng: np.random.Generator) -> str | None:
    """Autor favorito si existe."""
    if rng.random() < 0.65:
        autores = bib_config.get("signup_defaults", {}).get("autores_populares", ["Borges"])
        return str(rng.choice(autores))
    return None


def _generate_genero_favorito(bib_config: dict, rng: np.random.Generator) -> str | None:
    """Género favorito si existe."""
    if rng.random() < 0.70:
        generos = bib_config.get("signup_defaults", {}).get("generos_populares", ["Ficción"])
        return str(rng.choice(generos))
    return None


def _generate_tier_gasto(libros_leidos_total: int, bib_config: dict, rng: np.random.Generator) -> str:
    """Tier de gasto correlacionado con libros leídos."""
    dist = bib_config["field_distributions"]["tier_gasto"]
    values = dist["values"]   # ["Low", "Medium", "High"]
    weights = np.array(dist["weights"], dtype=float)

    # Skew: high readers → higher spend, low readers → lower spend
    if libros_leidos_total >= 20:
        weights = weights * np.array([0.5, 1.0, 2.0])
    elif libros_leidos_total <= 4:
        weights = weights * np.array([1.5, 1.0, 0.4])

    weights = weights / weights.sum()
    return str(rng.choice(values, p=weights))


def _generate_ingresos_app_mes(frecuencia: str, bib_config: dict, rng: np.random.Generator) -> int:
    """Sesiones/logins por mes correlacionadas con frecuencia de apertura."""
    params = bib_config["field_distributions"]["ingresos_app_mes"]["by_frecuencia"]
    p = params.get(frecuencia, params["monthly"])
    raw = rng.normal(p["mean"], p["std"])
    return int(np.clip(raw, p["min"], p["max"]))


def _generate_edad(bib_config: dict, rng: np.random.Generator) -> int:
    """Edad según distribución demográfica argentina de suscripciones."""
    brackets = bib_config["field_distributions"]["edad"]["brackets"]
    weights = np.array([b["weight"] for b in brackets], dtype=float)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(brackets), p=weights))
    lo, hi = brackets[idx]["range"]
    return int(rng.integers(lo, hi + 1))


def _generate_decline_code(
    bib_config: dict, rng: np.random.Generator
) -> str:
    """Código de rechazo sesgado a 51 (~50%)."""
    weights_dict = bib_config.get("decline_code_weights", {})
    if not weights_dict:
        codes_data = _load_decline_codes()
        weights_dict = {
            code: info.get("prevalencia_weight", 0.01)
            for code, info in codes_data.items()
        }

    codes = list(weights_dict.keys())
    weights = np.array([weights_dict[c] for c in codes])
    weights = weights / weights.sum()

    return str(rng.choice(codes, p=weights))


def _generate_attempt_number(rng: np.random.Generator) -> int:
    """Número de intento (1-15, sesgado a intentos tempranos)."""
    attempts = np.arange(1, 16)
    probs = 0.75 ** (attempts - 1)
    probs = probs / probs.sum()
    return int(rng.choice(attempts, p=probs))


def _generate_event_date(rng: np.random.Generator) -> str:
    """Fecha del evento (últimos 6 meses)."""
    start = date(2025, 12, 6)
    end = date(2026, 6, 6)
    span = (end - start).days
    event_date = start + timedelta(days=int(rng.integers(0, span + 1)))
    return event_date.isoformat()


def generate(
    n: int = 1000,
    output_path: Path | str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera dataset sintético de eventos de pago fallido para Biblioteca.

    Args:
        n: Cantidad de filas.
        output_path: Ruta de salida. Defaults a config.BIBLIOTECA_EVENTS_PATH.
        seed: Seed para reproducibilidad.

    Returns:
        DataFrame con todas las columnas generadas.
    """
    if output_path is None:
        output_path = config.DATA_DIR / "synthetic" / "biblioteca_events.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    bib_config = _load_biblioteca_config()

    filas = []
    for i in range(n):
        first_name = _generate_firstname(rng)
        last_name = _generate_lastname(rng)
        ciudad, state = _generate_ciudad_provincia(rng, bib_config)
        street_address, postcode = _generate_address(rng)
        phone = _generate_phone(rng)
        email = _generate_email(first_name, last_name, rng)
        cuit_dni = _generate_cuit_dni(rng)

        libros_leidos_total = _generate_libros_leidos(rng)
        libros_leidos_ultimos_3_meses = _generate_libros_3m(libros_leidos_total, rng)
        rating_promedio_dado = _generate_rating_promedio(rng)
        frecuencia_apertura_app = _generate_frecuencia_apertura(rng)
        ultimo_acceso_dias = _generate_ultimo_acceso_dias(rng)
        lista_deseos_activa = _generate_lista_deseos(rng)
        resenas_escritas = _generate_resenas_escritas(rng)
        autor_favorito = _generate_autor_favorito(bib_config, rng)
        genero_favorito = _generate_genero_favorito(bib_config, rng)

        tier_gasto = _generate_tier_gasto(libros_leidos_total, bib_config, rng)
        ingresos_app_mes = _generate_ingresos_app_mes(frecuencia_apertura_app, bib_config, rng)
        edad = _generate_edad(bib_config, rng)

        decline_code = _generate_decline_code(bib_config, rng)
        attempt_number = _generate_attempt_number(rng)
        event_date = _generate_event_date(rng)

        filas.append({
            "user_id": f"USR{i:07d}",
            "event_date": event_date,
            "first_name": first_name,
            "last_name": last_name,
            "country": "Argentina",
            "street_address": street_address,
            "ciudad": ciudad,
            "state": state,
            "postcode": postcode,
            "phone": phone,
            "email": email,
            "cuit_dni": cuit_dni,
            "payment_email": email,
            "libros_leidos_total": int(libros_leidos_total),
            "libros_leidos_ultimos_3_meses": int(libros_leidos_ultimos_3_meses),
            "rating_promedio_dado": float(round(rating_promedio_dado, 2)),
            "frecuencia_apertura_app": frecuencia_apertura_app,
            "ultimo_acceso_dias": int(ultimo_acceso_dias),
            "lista_deseos_activa": bool(lista_deseos_activa),
            "resenas_escritas": int(resenas_escritas),
            "autor_favorito": autor_favorito,
            "genero_favorito": genero_favorito,
            "tier_gasto": tier_gasto,
            "ingresos_app_mes": ingresos_app_mes,
            "edad": edad,
            "decline_code": decline_code,
            "attempt_number": attempt_number,
        })

    df = pd.DataFrame(filas)

    # Orden de columnas: signup, in-app behavior, transaction
    column_order = [
        "user_id", "event_date",
        "first_name", "last_name", "country", "street_address", "ciudad", "state",
        "postcode", "phone", "email", "cuit_dni", "payment_email",
        "libros_leidos_total", "libros_leidos_ultimos_3_meses", "rating_promedio_dado",
        "frecuencia_apertura_app", "ultimo_acceso_dias", "lista_deseos_activa",
        "resenas_escritas", "autor_favorito", "genero_favorito",
        "tier_gasto", "ingresos_app_mes", "edad",
        "decline_code", "attempt_number",
    ]
    df = df[column_order]

    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Imprime resumen de distribuciones."""
    print(f"\n--- Dataset Summary (N={len(df)}) ---")
    print(f"  Decline code 51 prevalence: {(df['decline_code'] == '51').mean():.3f}")
    print(f"  Top 3 decline codes:")
    for code, pct in df["decline_code"].value_counts(normalize=True).head(3).items():
        print(f"    {code}: {pct:.3f}")
    print(f"  Top 5 cities:")
    for city, pct in df["ciudad"].value_counts(normalize=True).head(5).items():
        print(f"    {city}: {pct:.3f}")
    print(f"  AMBA prevalence (Buenos Aires + CABA + La Plata): "
          f"{df[df['ciudad'].isin(['Buenos Aires', 'CABA', 'La Plata'])].shape[0] / len(df):.3f}")
    print(f"  Attempt distribution: {dict(df['attempt_number'].value_counts(normalize=True).round(3))}")
    print(f"  libro_leidos_total — mean: {df['libros_leidos_total'].mean():.1f}, "
          f"median: {df['libros_leidos_total'].median():.1f}")
    print(f"  ultimo_acceso_dias — mean: {df['ultimo_acceso_dias'].mean():.1f}, "
          f"median: {df['ultimo_acceso_dias'].median():.1f}")
    print(f"  tier_gasto — {dict(df['tier_gasto'].value_counts(normalize=True).round(3))}")
    print(f"  ingresos_app_mes — mean: {df['ingresos_app_mes'].mean():.1f}, "
          f"median: {df['ingresos_app_mes'].median():.1f}")
    print(f"  edad — mean: {df['edad'].mean():.1f}, "
          f"min: {df['edad'].min()}, max: {df['edad'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de dataset sintético para Vertical 1 (Biblioteca)."
    )
    parser.add_argument("--rows", type=int, default=1000, help="Cantidad de filas.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ruta de salida. Default: data/synthetic/biblioteca_events.csv",
    )
    args = parser.parse_args()

    df = generate(n=args.rows, output_path=args.out, seed=args.seed)
    _print_summary(df)
    print(f"  OK - Escribido: {args.out or 'data/synthetic/biblioteca_events.csv'}")
