# === recovery_actions.py ===
"""Resuelve la acción de recuperación para el usuario final según el código de rechazo.

La solución NO es siempre "actualizá tu medio de pago": depende del código.
  - Soft declines (51, 65, 05, 91...): se reintenta el cobro solo -> el usuario no hace nada.
  - Tarjeta vencida (54): hay que actualizar los datos.
  - Perdida/robada/inválida (41, 43, 57, 14): hay que cargar OTRO medio de pago.

Esto es un resolver liviano que lee data/decline_codes.json directamente (no depende
del modelo ML ni del context_builder). Devuelve el texto listo para el slot {accion_pago}.

Uso:
    from recovery_actions import resolver_accion
    accion = resolver_accion("51")   # -> {"tipo", "requiere_usuario", "texto", ...}
"""
from __future__ import annotations

import json
from typing import Any

import config

_DATA: dict | None = None


def _load() -> dict:
    global _DATA
    if _DATA is None:
        with open(config.DECLINE_CODES_PATH, encoding="utf-8") as f:
            _DATA = json.load(f)
    return _DATA


def resolver_accion(decline_code: str) -> dict[str, Any]:
    """Devuelve la acción de recuperación para un código de rechazo.

    Returns:
        {
            "decline_code": str,
            "descripcion": str,        # motivo legible (ej. "Fondos insuficientes")
            "tipo": str,               # reintento_auto | actualizar_datos | metodo_alternativo
            "requiere_usuario": bool,  # ¿el usuario tiene que hacer algo?
            "reintentable": bool,      # ¿se puede reintentar el cobro?
            "texto": str,              # mensaje para el usuario (slot {accion_pago})
        }
    """
    data = _load()
    codes = data.get("codes", {})
    acciones = data.get("acciones_usuario", {})
    code = str(decline_code).strip()

    # Normalizar: los códigos ISO son de 2 dígitos ("05", no "5"). Tolerar el cero perdido.
    if code not in codes and code.isdigit() and len(code) == 1:
        code = code.zfill(2)

    info = codes.get(code, {})
    best_action = info.get("best_action", "_default")
    accion = acciones.get(best_action) or acciones.get("_default", {})

    return {
        "decline_code": code,
        "descripcion": info.get("descripcion", "Pago rechazado"),
        "tipo": accion.get("tipo", "actualizar_datos"),
        "requiere_usuario": accion.get("requiere_usuario", True),
        "reintentable": not info.get("never_retry", False),
        "texto": accion.get("texto", acciones.get("_default", {}).get("texto", "")),
    }


if __name__ == "__main__":
    for c in ["51", "65", "05", "91", "54", "41", "57", "14", "99"]:
        a = resolver_accion(c)
        print(f"[{c}] {a['descripcion']:<35} tipo={a['tipo']:<18} "
              f"usuario={'sí' if a['requiere_usuario'] else 'no'}")
        print(f"      {a['texto']}")
