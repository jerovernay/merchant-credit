"""Calendario salarial argentino para Retenelo.

Lógica de fechas pura (solo stdlib) que el motor de recuperación y el generador de
datos usan para alinear los reintentos con los momentos en que los clientes
efectivamente tienen fondos. Las claves de ventana son strings y deben mantenerse en
sync con ``config.VENTANAS_REINTENTO``.

Ventanas clave:
  - quincena        -> el 15 (depósito de sueldo), ajustado a día hábil
  - fin_de_mes      -> último día hábil del mes
  - anses           -> fecha de cobro ANSES por terminación de DNI [PENDING calendario real]
  - aguinaldo       -> SAC: 30/06 (+4 días hábiles de gracia) y 18-23/12

Supuestos (documentados, sujetos a research):
  - Si el 15 cae en fin de semana/feriado, los empleadores suelen depositar el día
    hábil PREVIO, por eso ``fecha_quincena`` rueda hacia atrás.
  - Las ventanas "post_*" devuelven la próxima ocurrencia estrictamente POSTERIOR a la
    fecha dada.
  - ``FERIADOS`` cubre solo 2026; para otros años se usa solo la regla de fin de semana
    [PENDING feriados multi-año].
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

# --- Feriados nacionales 2026 (subconjunto de fecha fija; trasladables [PENDING]) ---
FERIADOS: set[date] = {
    date(2026, 1, 1),    # Año Nuevo
    date(2026, 2, 16),   # Carnaval
    date(2026, 2, 17),   # Carnaval
    date(2026, 3, 24),   # Día Nacional de la Memoria
    date(2026, 4, 2),    # Día del Veterano (Malvinas)
    date(2026, 4, 3),    # Viernes Santo
    date(2026, 5, 1),    # Día del Trabajador
    date(2026, 5, 25),   # Día de la Revolución de Mayo
    date(2026, 6, 17),   # Güemes [PENDING posible traslado]
    date(2026, 6, 20),   # Paso a la Inmortalidad de Belgrano
    date(2026, 7, 9),    # Día de la Independencia
    date(2026, 8, 17),   # Paso a la Inmortalidad de San Martín
    date(2026, 10, 12),  # Día del Respeto a la Diversidad Cultural [PENDING traslado]
    date(2026, 11, 20),  # Día de la Soberanía Nacional [PENDING traslado]
    date(2026, 12, 8),   # Inmaculada Concepción
    date(2026, 12, 25),  # Navidad
}

# ANSES: cuántas terminaciones de DNI se pagan por día hábil (modelo simplificado).
ANSES_TERMINACIONES_POR_DIA = 1  # [PENDING calendario oficial por terminación]


# --- Helpers de días hábiles ---
def es_dia_habil(fecha: date) -> bool:
    """True si ``fecha`` es día hábil (lun-vie y no feriado)."""
    return fecha.weekday() < 5 and fecha not in FERIADOS


def proximo_dia_habil(fecha: date) -> date:
    """Primer día hábil en o después de ``fecha``."""
    d = fecha
    while not es_dia_habil(d):
        d += timedelta(days=1)
    return d


def dia_habil_anterior(fecha: date) -> date:
    """Primer día hábil en o antes de ``fecha``."""
    d = fecha
    while not es_dia_habil(d):
        d -= timedelta(days=1)
    return d


def sumar_dias_habiles(fecha: date, n: int) -> date:
    """Suma ``n`` días hábiles a ``fecha`` (n >= 0); el día base no cuenta."""
    if n < 0:
        raise ValueError("n debe ser >= 0")
    d = fecha
    pasos = 0
    while pasos < n:
        d += timedelta(days=1)
        if es_dia_habil(d):
            pasos += 1
    return d


def _siguiente_mes(anio: int, mes: int) -> tuple[int, int]:
    """Devuelve (año, mes) del mes siguiente."""
    return (anio + 1, 1) if mes == 12 else (anio, mes + 1)


# --- Anclas de ventanas salariales ---
def ultimo_dia_habil_mes(anio: int, mes: int) -> date:
    """Último día hábil del mes (ancla de 'fin de mes')."""
    ultimo_dia = calendar.monthrange(anio, mes)[1]
    return dia_habil_anterior(date(anio, mes, ultimo_dia))


def fecha_quincena(anio: int, mes: int) -> date:
    """Fecha de depósito de quincena (~15), ajustada al día hábil previo."""
    return dia_habil_anterior(date(anio, mes, 15))


def fechas_anses(dni_terminacion: int, anio: int, mes: int) -> date:
    """Fecha estimada de cobro ANSES según terminación de DNI (0-9). [PENDING calendario real]."""
    if not 0 <= dni_terminacion <= 9:
        raise ValueError("dni_terminacion debe estar entre 0 y 9")
    inicio = proximo_dia_habil(date(anio, mes, 1))
    offset = dni_terminacion // ANSES_TERMINACIONES_POR_DIA
    return sumar_dias_habiles(inicio, offset)


def ventanas_aguinaldo(anio: int) -> list[tuple[date, date]]:
    """Ventanas de cobro de aguinaldo (SAC) del año: [(inicio, fin), ...]."""
    sac1_inicio = date(anio, 6, 30)
    sac1_fin = sumar_dias_habiles(sac1_inicio, 4)  # 4 días hábiles de gracia
    sac2_inicio = date(anio, 12, 18)
    sac2_fin = date(anio, 12, 23)
    return [(sac1_inicio, sac1_fin), (sac2_inicio, sac2_fin)]


def es_mes_aguinaldo(fecha: date) -> bool:
    """True si el mes es de pago de aguinaldo (junio o diciembre)."""
    return fecha.month in (6, 12)


# --- Features temporales (para generador y modelo) ---
def dias_hasta_quincena(fecha: date) -> int:
    """Días hasta la próxima quincena (>= 0)."""
    q = fecha_quincena(fecha.year, fecha.month)
    if q < fecha:
        anio, mes = _siguiente_mes(fecha.year, fecha.month)
        q = fecha_quincena(anio, mes)
    return (q - fecha).days


def dias_hasta_fin_de_mes(fecha: date) -> int:
    """Días hasta el último día hábil del mes (>= 0)."""
    fdm = ultimo_dia_habil_mes(fecha.year, fecha.month)
    if fdm < fecha:
        anio, mes = _siguiente_mes(fecha.year, fecha.month)
        fdm = ultimo_dia_habil_mes(anio, mes)
    return (fdm - fecha).days


def proxima_ventana_cobro(fecha: date, ventana_key: str) -> date:
    """Próxima fecha concreta de la ventana indicada, estrictamente posterior a ``fecha``.

    ``ventana_key`` debe ser una de ``config.VENTANAS_REINTENTO`` (o un alias de cobro de
    arquetipo: 'quincena', 'fin_de_mes', 'anses'). Si la clave no se reconoce, devuelve
    el próximo día hábil como fallback seguro.
    """
    key = ventana_key.lower()

    if key == "inmediata":
        return proximo_dia_habil(fecha + timedelta(days=1))

    if key in ("quincena", "post_quincena"):
        anio, mes = fecha.year, fecha.month
        q = fecha_quincena(anio, mes)
        if q <= fecha:
            anio, mes = _siguiente_mes(anio, mes)
            q = fecha_quincena(anio, mes)
        return q

    if key == "fin_de_mes":
        anio, mes = fecha.year, fecha.month
        fdm = ultimo_dia_habil_mes(anio, mes)
        if fdm <= fecha:
            anio, mes = _siguiente_mes(anio, mes)
            fdm = ultimo_dia_habil_mes(anio, mes)
        return fdm

    if key == "anses":
        anio, mes = fecha.year, fecha.month
        a = fechas_anses(5, anio, mes)  # terminación representativa [PENDING DNI real]
        if a <= fecha:
            anio, mes = _siguiente_mes(anio, mes)
            a = fechas_anses(5, anio, mes)
        return a

    if key == "post_aguinaldo":
        for anio in (fecha.year, fecha.year + 1):
            for inicio, _fin in ventanas_aguinaldo(anio):
                objetivo = proximo_dia_habil(inicio)
                if objetivo > fecha:
                    return objetivo
        return proximo_dia_habil(fecha + timedelta(days=1))  # fallback (no debería ocurrir)

    if key == "post_fecha_resumen":
        # Vencimiento de resumen ~ día 10 del mes.
        anio, mes = fecha.year, fecha.month
        r = proximo_dia_habil(date(anio, mes, 10))
        if r <= fecha:
            anio, mes = _siguiente_mes(anio, mes)
            r = proximo_dia_habil(date(anio, mes, 10))
        return r

    return proximo_dia_habil(fecha + timedelta(days=1))


def features_temporales(fecha: date) -> dict:
    """Bundle de features temporales para el generador y el modelo (valores ML-ready)."""
    return {
        "day_of_month": fecha.day,
        "days_to_quincena": dias_hasta_quincena(fecha),
        "days_to_fin_de_mes": dias_hasta_fin_de_mes(fecha),
        "is_aguinaldo_month": int(es_mes_aguinaldo(fecha)),
        "anses_pay_flag": int(fecha.day <= 12),  # ventana aprox. de pago ANSES [PENDING]
        "es_dia_habil": int(es_dia_habil(fecha)),
    }
