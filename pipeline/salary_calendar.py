# === salary_calendar.py ===
"""Re-exporta el calendario salarial argentino desde data/salary_calendar.py.

Permite importar directamente con ``import salary_calendar`` desde la raíz del proyecto.
Agrega ``get_next_payday`` como función pública canónica que unifica la lógica de
quincena, fin de mes y ANSES según el perfil del usuario.
"""
from __future__ import annotations

from datetime import date

# Re-export everything from the implementation module.
from data.salary_calendar import (  # noqa: F401
    FERIADOS,
    es_dia_habil,
    proximo_dia_habil,
    dia_habil_anterior,
    sumar_dias_habiles,
    ultimo_dia_habil_mes,
    fecha_quincena,
    fechas_anses,
    ventanas_aguinaldo,
    es_mes_aguinaldo,
    dias_hasta_quincena,
    dias_hasta_fin_de_mes,
    proxima_ventana_cobro,
    features_temporales,
)


def get_next_payday(dni_last_digit: int, reference_date: date) -> date:
    """Devuelve la próxima fecha de cobro estimada según la terminación del DNI.

    Combina la lógica de ANSES (por terminación de DNI) con la ventana de quincena
    y fin de mes, y devuelve la que cae primero luego de ``reference_date``.

    Args:
        dni_last_digit: Último dígito del DNI (0-9).
        reference_date: Fecha base desde la que calcular.

    Returns:
        La próxima fecha de cobro estimada (date).
    """
    if not 0 <= dni_last_digit <= 9:
        raise ValueError("dni_last_digit debe estar entre 0 y 9.")

    candidates: list[date] = []

    anses_date = fechas_anses(dni_last_digit, reference_date.year, reference_date.month)
    if anses_date > reference_date:
        candidates.append(anses_date)
    else:
        # Try next month
        next_m = reference_date.month % 12 + 1
        next_y = reference_date.year + (1 if reference_date.month == 12 else 0)
        candidates.append(fechas_anses(dni_last_digit, next_y, next_m))

    q = proxima_ventana_cobro(reference_date, "quincena")
    candidates.append(q)

    fdm = proxima_ventana_cobro(reference_date, "fin_de_mes")
    candidates.append(fdm)

    return min(candidates)
