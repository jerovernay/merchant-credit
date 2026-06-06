# === simulator.py ===
"""Simulador de ROI para la demo B2B de Retenelo.

Calcula el retorno económico que obtiene una empresa cliente al usar Retenelo
vs no hacer nada ante pagos fallidos. Sin dependencias externas.

Uso:
    from simulator import calculate_roi
    roi = calculate_roi(n_failed_payments=500, avg_ticket_ars=8000)

    python simulator.py  # imprime demo con valores por defecto
"""
from __future__ import annotations

import config


def calculate_roi(
    n_failed_payments: int,
    avg_ticket_ars: float,
    recovery_rate: float = config.ROI_DEFAULTS["recovery_rate"],
    commission: float = config.ROI_DEFAULTS["commission_pct"],
    retention_months: int = config.ROI_DEFAULTS["retention_months"],
) -> dict[str, float]:
    """Calcula el ROI de usar Retenelo para recuperar pagos fallidos.

    Args:
        n_failed_payments: Cantidad de pagos fallidos en el período.
        avg_ticket_ars: Ticket promedio en ARS por suscripción.
        recovery_rate: Tasa de recuperación estimada (0.0 - 1.0). Default 40%.
        commission: Comisión de Retenelo sobre ingresos recuperados (0.0 - 1.0). Default 12.5%.
        retention_months: Meses promedio de retención adicional del suscriptor recuperado.

    Returns:
        {
            "pagos_fallidos":          int,
            "monto_en_riesgo_ars":     float,   # total ARS en riesgo
            "pagos_recuperados":       float,   # cantidad estimada recuperada
            "revenue_recuperado_ars":  float,   # ARS recuperados
            "comision_retenelo_ars":   float,   # lo que paga la empresa a Retenelo
            "ingreso_neto_cliente_ars":float,   # ARS que retiene el cliente
            "ltv_uplift_ars":          float,   # valor del ciclo de vida adicional
            "roi_vs_no_hacer_nada":    float,   # ratio ingreso_neto / monto_en_riesgo
            "roi_pct":                 float,   # roi_vs_no_hacer_nada en porcentaje
        }
    """
    monto_en_riesgo = n_failed_payments * avg_ticket_ars
    pagos_recuperados = n_failed_payments * recovery_rate
    revenue_recuperado = pagos_recuperados * avg_ticket_ars
    comision = revenue_recuperado * commission
    ingreso_neto = revenue_recuperado - comision

    # LTV uplift: cada suscriptor recuperado genera (retention_months - 1) meses extra.
    ltv_uplift = pagos_recuperados * avg_ticket_ars * max(0, retention_months - 1)

    roi_ratio = ingreso_neto / monto_en_riesgo if monto_en_riesgo > 0 else 0.0

    return {
        "pagos_fallidos": float(n_failed_payments),
        "monto_en_riesgo_ars": round(monto_en_riesgo, 2),
        "pagos_recuperados": round(pagos_recuperados, 1),
        "revenue_recuperado_ars": round(revenue_recuperado, 2),
        "comision_retenelo_ars": round(comision, 2),
        "ingreso_neto_cliente_ars": round(ingreso_neto, 2),
        "ltv_uplift_ars": round(ltv_uplift, 2),
        "roi_vs_no_hacer_nada": round(roi_ratio, 4),
        "roi_pct": round(roi_ratio * 100, 2),
    }


def _demo() -> None:
    """Imprime un ejemplo de ROI con los defaults de config.py."""
    defaults = config.ROI_DEFAULTS
    n = int(defaults["subscriber_count"] * defaults["monthly_failure_rate"])
    ticket = defaults["arpu_ars"]

    roi = calculate_roi(
        n_failed_payments=n,
        avg_ticket_ars=ticket,
        recovery_rate=defaults["recovery_rate"],
        commission=defaults["commission_pct"],
        retention_months=defaults["retention_months"],
    )

    print("\n--- Simulador ROI Retenelo (demo) ---")
    print(f"  Pagos fallidos en el período : {int(roi['pagos_fallidos'])}")
    print(f"  Monto en riesgo              : ARS {roi['monto_en_riesgo_ars']:,.0f}")
    print(f"  Pagos estimados recuperados  : {roi['pagos_recuperados']:.0f}")
    print(f"  Revenue recuperado           : ARS {roi['revenue_recuperado_ars']:,.0f}")
    print(f"  Comisión Retenelo (12.5%)    : ARS {roi['comision_retenelo_ars']:,.0f}")
    print(f"  Ingreso neto cliente         : ARS {roi['ingreso_neto_cliente_ars']:,.0f}")
    print(f"  LTV uplift ({defaults['retention_months']} meses)          : ARS {roi['ltv_uplift_ars']:,.0f}")
    print(f"  ROI vs no hacer nada         : {roi['roi_pct']:.1f}%")


if __name__ == "__main__":
    _demo()
