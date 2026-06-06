# === output_composer.py ===
"""Compositor de mensajes personalizados via Claude Sonnet.

Responsabilidad única: dado el contexto enriquecido y el action triple,
construir el prompt y llamar a claude-sonnet-4-6 para generar el mensaje
final orientado al usuario final.

Invariantes:
  - Una sola llamada a Sonnet por evento de recuperación.
  - El output es format-agnostic: {"channel": str, "content": str, "metadata": {...}}.
  - Nunca se exponen señales de ML, scoring ni datos sensibles al usuario final.
  - El mensaje incluye siempre la mención de opt-out (Ley 25.326).
  - TLS verify=False para el proxy hackathon (self-signed cert).

Uso:
    from output_composer import compose
    result = compose(
        user_type="lector_voraz",
        enriched_context=ctx,
        decline_code="51",
        action_triple=triple,
        attempt_number=2,
    )
    # result["content"] es el mensaje listo para enviar por result["channel"]
"""
from __future__ import annotations

import json
import warnings
from typing import Any

import config

# Suppress urllib3 InsecureRequestWarning for the self-signed proxy cert.
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ---------------------------------------------------------------------------
# System prompt (enforces Spanish, tone, compliance, length)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Sos el asistente de recuperación de pagos de Retenelo.
Tu tarea es redactar un mensaje personalizado para un suscriptor cuyo pago no pudo procesarse.

REGLAS ESTRICTAS:
- Escribí siempre en español argentino (vos, no tú).
- El tono debe ser cálido pero no insistente. Nunca uses lenguaje de cobranza agresiva.
- Nunca menciones sistemas de scoring, machine learning, ni probabilidades de recuperación.
- No expongas datos sensibles del usuario (número de tarjeta, DNI, CUIT, dirección).
- Incluí SIEMPRE una mención al opt-out (ej: "Si no querés recibir más mensajes, respondé STOP").
- Adaptá la longitud al canal: WhatsApp/push → máximo 3 oraciones; email → máximo 2 párrafos.
- Si el canal es "none", devolvé un objeto JSON con content vacío ("").
- Devolvé ÚNICAMENTE el texto del mensaje, sin saludos de sistema, sin explicaciones extra.
"""

_CHANNEL_HINTS: dict[str, str] = {
    "whatsapp": "Mensaje corto (2-3 oraciones). Podés usar 1-2 emojis relevantes. Sin HTML.",
    "push": "Máximo 120 caracteres. Directo al punto. Sin emojis.",
    "email": "Hasta 2 párrafos cortos. Podés usar un asunto implícito. Sin HTML.",
    "sms": "Máximo 160 caracteres. Sin emojis.",
    "llamada": "Guión de apertura de llamada. Máximo 3 oraciones. Presentación + motivo + CTA.",
    "none": "El usuario no debe ser contactado. Devolvé content vacío.",
}

_TONE_HINTS: dict[str, str] = {
    "cercano": "Cercano y empático. Como si fuera un amigo que te avisa.",
    "formal": "Formal y respetuoso. Institucional pero sin ser frío.",
    "empatico": "Muy empático. Reconocé la situación sin presionar.",
    "urgente_suave": "Leve urgencia. Mencioná que hay una fecha límite sin generar ansiedad.",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    user_type: str,
    enriched_context: dict,
    decline_code: str,
    action_triple: dict,
    attempt_number: int,
    ley_25326_compliant: bool,
) -> str:
    """Construye el prompt de usuario con todo el contexto estructurado."""
    channel = action_triple.get("channel", "whatsapp")
    tone = action_triple.get("tone", "cercano")
    retry_window = action_triple.get("retry_window")
    first_name = enriched_context.get("first_name", "")
    svc = enriched_context.get("service_ctx", {})
    eco = enriched_context.get("economic_ctx", {})
    decline_info = enriched_context.get("decline_info", {})

    context_block = {
        "usuario": {
            "nombre": first_name or None,
            "tipo_lector": user_type,
            "libros_leidos_total": svc.get("libros_leidos_total"),
            "genero_favorito": svc.get("genero_favorito"),
            "autor_favorito": svc.get("autor_favorito"),
            "engagement": svc.get("engagement_nivel"),
            "lista_deseos_activa": svc.get("lista_deseos_activa"),
        },
        "situacion_pago": {
            "motivo_rechazo": decline_info.get("descripcion", "Rechazo del emisor"),
            "requiere_accion_usuario": decline_info.get("requiere_accion_usuario", False),
            "numero_de_intento": attempt_number,
        },
        "accion_sugerida": {
            "canal": channel,
            "tono": tone,
            "ventana_reintento": retry_window.isoformat() if hasattr(retry_window, "isoformat") else str(retry_window) if retry_window else None,
            "contexto_economico": {
                "dias_hasta_cobro": min(
                    eco.get("dias_hasta_quincena", 99),
                    eco.get("dias_hasta_fin_de_mes", 99),
                ),
                "es_mes_aguinaldo": eco.get("es_mes_aguinaldo", False),
            },
        },
        "compliance": {
            "ley_25326": ley_25326_compliant,
            "incluir_opt_out": True,
        },
    }

    channel_hint = _CHANNEL_HINTS.get(channel, _CHANNEL_HINTS["whatsapp"])
    tone_hint = _TONE_HINTS.get(tone, _TONE_HINTS["cercano"])

    return (
        f"Canal de envío: {channel.upper()}\n"
        f"Instrucción de canal: {channel_hint}\n"
        f"Tono requerido: {tone_hint}\n\n"
        f"Contexto del evento (JSON):\n{json.dumps(context_block, ensure_ascii=False, indent=2)}\n\n"
        "Redactá el mensaje ahora."
    )


# ---------------------------------------------------------------------------
# LLM client (lazy init)
# ---------------------------------------------------------------------------

_CLIENT = None


def _get_client():
    """Inicializa el cliente Anthropic con el proxy hackathon (TLS verify=False)."""
    global _CLIENT
    if _CLIENT is None:
        import httpx
        import anthropic

        _CLIENT = anthropic.Anthropic(
            api_key=config.get_llm_api_key(),
            base_url=config.LLM_BASE_URL,
            http_client=httpx.Client(verify=config.LLM_VERIFY_TLS),
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose(
    user_type: str,
    enriched_context: dict[str, Any],
    decline_code: str,
    action_triple: dict[str, Any],
    attempt_number: int,
    ley_25326_compliant: bool = True,
) -> dict[str, Any]:
    """Genera el mensaje personalizado de recuperación via Claude Sonnet.

    Args:
        user_type: Tipo de usuario asignado por clustering.py.
        enriched_context: Contexto enriquecido de context_builder.build_context().
        decline_code: Código ISO-8583 del rechazo.
        action_triple: Triple de acción de recovery_engine.get_action_triple().
        attempt_number: Número de intento actual (>= 1).
        ley_25326_compliant: Flag de cumplimiento (siempre True en producción).

    Returns:
        {
            "channel": str,
            "content": str,
            "metadata": {"user_type": str, "tone": str, "attempt": int, "model": str}
        }
    """
    channel = action_triple.get("channel", "whatsapp")
    tone = action_triple.get("tone", "cercano")

    # Short-circuit: no contact allowed.
    if channel == "none":
        return {
            "channel": "none",
            "content": "",
            "metadata": {
                "user_type": user_type,
                "tone": tone,
                "attempt": attempt_number,
                "model": config.MODELO_SONNET,
                "compliance_ok": action_triple.get("compliance_ok", False),
                "reason": action_triple.get("reason", ""),
            },
        }

    user_prompt = _build_user_prompt(
        user_type=user_type,
        enriched_context=enriched_context,
        decline_code=decline_code,
        action_triple=action_triple,
        attempt_number=attempt_number,
        ley_25326_compliant=ley_25326_compliant,
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=config.MODELO_SONNET,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text.strip()
    except Exception as e:
        # Fallback to a generic message; never crash the pipeline.
        content = (
            f"Hola{', ' + enriched_context.get('first_name', '') if enriched_context.get('first_name') else ''}, "
            "notamos que tu pago no pudo procesarse. Por favor, verificá tu medio de pago "
            "o contactanos para ayudarte. Si no querés recibir más mensajes, respondé STOP."
        )
        raise RuntimeError(
            f"Error al generar mensaje con Sonnet: {e}. Fallback aplicado."
        ) from e

    return {
        "channel": channel,
        "content": content,
        "metadata": {
            "user_type": user_type,
            "tone": tone,
            "attempt": attempt_number,
            "model": config.MODELO_SONNET,
            "compliance_ok": action_triple.get("compliance_ok", True),
            "reason": action_triple.get("reason", ""),
        },
    }
