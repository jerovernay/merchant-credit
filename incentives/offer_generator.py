# === offer_generator.py ===
"""Composer de ofertas personalizadas por cluster (Capas B y C del pipeline).

Reparto de responsabilidades:
  - El LLM decide el LENGUAJE y la SELECCIÓN (qué gancho usar, qué incentivo
    elegir del catálogo, cómo redactar el mensaje con slots por usuario).
  - El CÓDIGO es dueño de los NÚMEROS (costo del incentivo, LTV, margen) y aplica
    un guardrail económico: el incentivo no puede costar más que un % del LTV
    recuperado. El modelo nunca inventa precios.

Flujo:
  1. cluster_profiler.build_cluster_profiles()  -> perfil real del cluster (Capa A)
  2. generate_offer(cluster_id, perfil)          -> oferta estructurada + economía
  3. personalize_offer(template, user)           -> rellena {nombre}, {autor_favorito}...
  4. generate_all_offers()                        -> genera y cachea las 5 ofertas

El demo NO llama al LLM en vivo: usa el cache pregenerado y solo rellena slots.

Uso:
    python offer_generator.py            # genera y cachea (correr antes del demo)
    python offer_generator.py --force    # regenera aunque exista el cache
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import config
from incentives.offer_matcher import recomendar_oferta_personal
from pipeline.recovery_actions import resolver_accion

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

OFFERS_CACHE_PATH = Path("data/offers_cache.json")

# Slots que el LLM puede usar en el template y que rellenamos por usuario.
#   {nombre}          -> nombre del usuario
#   {autor_favorito}  -> autor que más lee (derivado de su historial)
#   {oferta_personal} -> oferta CONCRETA del catálogo (libro/evento), la resuelve el matcher
#   {accion_pago}     -> qué hacer según el código de rechazo (no siempre "actualizá la tarjeta")
PLACEHOLDERS_PERMITIDOS = ["{nombre}", "{autor_favorito}", "{oferta_personal}", "{accion_pago}"]

_SYSTEM_PROMPT = """Sos el especialista en retención de Retenelo, plataforma B2B que recupera
suscriptores para servicios de suscripción de libros en Argentina.

Tu tarea: para un SEGMENTO de usuarios, escribir UN template de mensaje de re-enganche, breve
y específico, y elegir UN incentivo del catálogo que sea rentable. El template se rellena por
usuario con datos reales.

FILOSOFÍA DEL MENSAJE (no negociable):
- El mensaje es 100% una OFERTA positiva. El usuario NO debe sentir que hay un problema.
- Estructura obligatoria, en este orden:
    1) Apertura corta, positiva y personalizada con {nombre} (ej: "Solo para vos, {nombre}:",
       "No te lo pierdas, {nombre}:", "Pensado para vos, {nombre}:").
    2) La oferta concreta como gancho ({oferta_personal}).
    3) Cierre: {accion_pago} — el ÚLTIMO paso para reclamar la oferta. Recién ahí el usuario
       se entera del paso de pago, en clave de "cómo lo activo", nunca de "algo falló".
- PROHIBIDO mencionar el motivo del rechazo o sugerir un problema. PALABRAS PROHIBIDAS:
  "volvé/volver", "reintentar", "se acredita/procesa", "vencida", "no pudimos", "extrañar",
  "perdés el acceso". Nada que insinúe ausencia o falla.

CÓMO PENSAR LA OFERTA:
- Mirá los intereses reales del segmento (autores y géneros más leídos) y las noticias.
- Elegí el gancho más persuasivo. Preferí un autor/género dominante por sobre una noticia.
- Elegí UN incentivo del catálogo según cuánto LEE el segmento:
    * Segmentos que leen mucho (voraz, fiel) o que hay que recuperar (inactivo): REGALÁ un
      libro completo ("libro_regalo"): se valora y se consume.
    * Segmentos que leen poco (casual, coleccionista): incentivo digital marginal (club,
      pack, acceso anticipado): el libro regalado se desperdiciaría.
    * "mes_gratis" es el más caro: solo con alto LTV.

REGLAS DEL MENSAJE (template) — estrictas:
- Español argentino (vos, no tú). Máximo 2 oraciones. Positivo y directo.
- Tono según el segmento (te lo indico).
- El mensaje es un TEMPLATE con placeholders. Placeholders permitidos EXACTOS:
    {nombre}          -> nombre de la persona (va en la apertura)
    {autor_favorito}  -> el autor que esa persona más lee
    {oferta_personal} -> la oferta CONCRETA (un libro/evento). Ya viene resuelta por el
                         sistema; vos solo la introducís (ej: "te regalamos {oferta_personal}").
                         NO describas el libro/evento vos mismo.
    {accion_pago}     -> el paso de pago para activar. SIEMPRE cerrá el mensaje con esta frase.
- CRÍTICO 1 (anti-alucinación): NUNCA escribas un autor, título, evento, fecha o precio
  concretos. Esos datos vienen SOLO de {oferta_personal} o {autor_favorito}. Inventar un
  libro o evento está terminantemente prohibido.
    PROHIBIDO: "te regalamos Rayuela de Cortázar"
    CORRECTO:  "te regalamos {oferta_personal}"
- CRÍTICO 2 (acción según el rechazo): NO escribas vos la instrucción de pago. La inyecta
  {accion_pago} según el código de cada usuario. Tu template SIEMPRE termina con {accion_pago}.
- Incluí {nombre}, {oferta_personal} y {accion_pago} sí o sí. {autor_favorito} si suma.
- No menciones scoring, machine learning, probabilidades ni datos sensibles.

FORMATO DE SALIDA — devolvé ÚNICAMENTE un JSON válido, sin texto extra:
{
  "tipo_gancho": "autor" | "genero" | "noticia",
  "gancho": "descripción corta del gancho elegido",
  "incentivo_id": "<id exacto del catálogo>",
  "mensaje_template": "texto con placeholders (incluye {oferta_personal} y termina en {accion_pago})",
  "justificacion": "por qué este gancho + incentivo para este segmento (1 oración)"
}"""


# ---------------------------------------------------------------------------
# Carga de configs
# ---------------------------------------------------------------------------

def _load_clusters() -> dict:
    with open(config.CLUSTERS_PATH, encoding="utf-8") as f:
        return json.load(f)["clusters"]


def _load_incentivos() -> dict:
    with open(config.INCENTIVOS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Capa B — economía del incentivo (determinística, el LLM no toca números)
# ---------------------------------------------------------------------------

def costo_incentivo_ars(incentivo: dict, arpu: float) -> float:
    """Costo por usuario recuperado de un incentivo, según su tipo de costo."""
    tipo = incentivo.get("tipo_costo", "marginal")
    if tipo == "marginal" or tipo == "fijo":
        return float(incentivo.get("costo_estimado_ars", 0))
    if tipo == "porcentual_libro":
        return round(config.PRECIO_LIBRO_PROMEDIO_ARS * float(incentivo.get("descuento_pct", 0)), 0)
    if tipo == "arpu":
        return float(arpu)
    return 0.0


def calcular_economia(incentivo_id: str, incentivos: dict, arpu: float) -> dict[str, Any]:
    """Calcula la economía de una oferta y si pasa el guardrail de margen.

    LTV recuperado = ARPU * meses de retención.
    Guardrail: el costo del incentivo no puede superar MARGEN_INCENTIVO_MAX del LTV.
    """
    inc = incentivos["incentivos"][incentivo_id]
    ltv = arpu * config.MESES_RETENCION
    costo = costo_incentivo_ars(inc, arpu)
    ratio = (costo / ltv) if ltv > 0 else 0.0
    dentro = ratio <= config.MARGEN_INCENTIVO_MAX

    return {
        "incentivo_id": incentivo_id,
        "incentivo_nombre": inc["nombre"],
        "valor_percibido": inc.get("valor_percibido", "medio"),
        "arpu_ars": round(arpu, 0),
        "ltv_recuperado_ars": round(ltv, 0),
        "costo_incentivo_ars": round(costo, 0),
        "margen_por_recuperado_ars": round(ltv - costo, 0),
        "ratio_costo_ltv": round(ratio, 3),
        "dentro_guardrail": dentro,
    }


def _aplicar_guardrail(incentivo_id: str, incentivos: dict, arpu: float) -> tuple[str, dict, bool]:
    """Si el incentivo elegido supera el guardrail, lo baja al fallback marginal.

    Returns: (incentivo_id_final, economia, fue_ajustado)
    """
    economia = calcular_economia(incentivo_id, incentivos, arpu)
    if economia["dentro_guardrail"]:
        return incentivo_id, economia, False

    fallback_id = incentivos.get("incentivo_fallback", "acceso_anticipado")
    economia_fb = calcular_economia(fallback_id, incentivos, arpu)
    return fallback_id, economia_fb, True


# ---------------------------------------------------------------------------
# Cliente LLM (proxy OpenAI del hackathon)
# ---------------------------------------------------------------------------

def _get_client():
    from openai import OpenAI
    return OpenAI(api_key=config.get_llm_api_key())


def _build_prompt(cluster_data: dict, perfil: dict | None, incentivos: dict) -> str:
    """Arma el prompt del usuario con intereses reales, noticias y catálogo."""
    # Intereses reales del segmento (Capa A)
    if perfil:
        autores = ", ".join(f"{a['valor']} ({a['n']} lectores)" for a in perfil.get("top_autores", [])) or "sin datos"
        generos = ", ".join(f"{g['valor']} ({g['n']} lectores)" for g in perfil.get("top_generos", [])) or "sin datos"
        intereses = (
            f"Autores más leídos del segmento: {autores}\n"
            f"Géneros más leídos del segmento: {generos}\n"
            f"Libros por mes (promedio): {perfil.get('libros_mes_prom', '-')}\n"
            f"ARPU del segmento: ARS {perfil.get('arpu_ars', '-'):.0f}"
        )
    else:
        intereses = "(sin datos agregados del segmento — usá la descripción y las noticias)"

    # Noticias del cluster
    noticias = cluster_data.get("noticias_relevantes", [])
    noticias_texto = "\n".join(f"  - [{n['fuente']}] {n['titulo']}" for n in noticias)

    # Catálogo de incentivos
    cat_lineas = []
    for inc_id, inc in incentivos["incentivos"].items():
        costo = inc.get("costo_estimado_ars", 0)
        if inc.get("tipo_costo") == "porcentual_libro":
            costo_txt = f"~ARS {config.PRECIO_LIBRO_PROMEDIO_ARS * inc['descuento_pct']:.0f} (descuento)"
        elif inc.get("tipo_costo") == "arpu":
            costo_txt = "= 1 mes de suscripción (CARO)"
        elif costo == 0:
            costo_txt = "~ARS 0 (marginal)"
        else:
            costo_txt = f"~ARS {costo:.0f}"
        cat_lineas.append(f"  - {inc_id}: {inc['nombre']} — costo {costo_txt}, valor percibido {inc.get('valor_percibido')}")
    catalogo_texto = "\n".join(cat_lineas)

    return f"""SEGMENTO: {cluster_data['nombre']}
Descripción: {cluster_data['descripcion']}
Tono requerido: {cluster_data['tono_oferta']}

INTERESES REALES DEL SEGMENTO (datos de comportamiento):
{intereses}

NOTICIAS Y EVENTOS DEL MOMENTO:
{noticias_texto}

CATÁLOGO DE INCENTIVOS (elegí UN incentivo_id de esta lista):
{catalogo_texto}

CONTEXTO: el pago de su suscripción no se pudo procesar; necesitan actualizar el medio
de pago para mantener el acceso.

Diseñá la oferta y devolvé el JSON."""


# ---------------------------------------------------------------------------
# Capa C — composer
# ---------------------------------------------------------------------------

def _fallback_offer(cluster_id: str, cluster_data: dict, incentivos: dict, arpu: float) -> dict[str, Any]:
    """Oferta de respaldo si el LLM falla. El demo nunca queda en blanco."""
    fallback_id = incentivos.get("incentivo_fallback", "acceso_anticipado")
    economia = calcular_economia(fallback_id, incentivos, arpu)
    return {
        "tipo_gancho": "genero",
        "gancho": "interés general del segmento",
        "incentivo_id": fallback_id,
        "incentivo_ajustado": False,
        "mensaje_template": (
            "Solo para vos, {nombre}: te regalamos {oferta_personal}. {accion_pago}"
        ),
        "justificacion": "Respaldo genérico con incentivo de costo marginal.",
        "economia": economia,
    }


def generate_offer(
    cluster_id: str,
    perfil: dict | None = None,
) -> dict[str, Any]:
    """Genera una oferta estructurada para un cluster (lenguaje del LLM + economía del código).

    Args:
        cluster_id: ID del cluster (ej: "lector_fiel").
        perfil: Perfil agregado del cluster (de cluster_profiler). Si None, usa solo
            la descripción y las noticias del clusters.json.

    Returns:
        dict con tipo_gancho, gancho, incentivo_id, mensaje_template, justificacion,
        economia (costo/LTV/margen/guardrail), e incentivo_ajustado (bool).
    """
    clusters = _load_clusters()
    cluster_data = clusters.get(cluster_id)
    if not cluster_data:
        raise ValueError(f"Cluster desconocido: {cluster_id}. Disponibles: {list(clusters.keys())}")

    incentivos = _load_incentivos()
    arpu = float((perfil or {}).get("arpu_ars") or config.ARPU_POR_TIER_ARS["Medium"])

    prompt = _build_prompt(cluster_data, perfil, incentivos)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL_OFERTAS,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[offer_generator] Error con el LLM: {e}. Usando fallback.")
        result = _fallback_offer(cluster_id, cluster_data, incentivos, arpu)
        result.update({
            "cluster_id": cluster_id,
            "cluster_nombre": cluster_data["nombre"],
            "model": "fallback",
            "generated_at": date.today().isoformat(),
        })
        return result

    # Validar incentivo elegido; si no existe, ir al fallback del catálogo.
    incentivo_id = data.get("incentivo_id")
    if incentivo_id not in incentivos["incentivos"]:
        incentivo_id = incentivos.get("incentivo_fallback", "acceso_anticipado")

    # Guardrail económico: el código manda sobre la elección del modelo.
    incentivo_final, economia, ajustado = _aplicar_guardrail(incentivo_id, incentivos, arpu)

    # Garantía de slots: el mensaje DEBE poder ofrecer algo concreto y cerrar con la
    # acción de pago. Si el modelo no los incluyó, los inyectamos (robustez del demo).
    template = _asegurar_slots(data.get("mensaje_template", ""))

    return {
        "cluster_id": cluster_id,
        "cluster_nombre": cluster_data["nombre"],
        "tipo_gancho": data.get("tipo_gancho", "noticia"),
        "gancho": data.get("gancho", ""),
        "incentivo_id": incentivo_final,
        "incentivo_ajustado": ajustado,
        "mensaje_template": template,
        "justificacion": data.get("justificacion", ""),
        "economia": economia,
        "model": config.OPENAI_MODEL_OFERTAS,
        "generated_at": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Relleno por usuario (lo único que corre "en vivo" en el demo)
# ---------------------------------------------------------------------------

class _SafeDict(dict):
    """dict para str.format_map que tolera placeholders desconocidos."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _clean(value: Any) -> str:
    """Normaliza un valor del CSV; '' si es NaN/vacío."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s


def _asegurar_slots(template: str) -> str:
    """Garantiza que el template ofrezca algo concreto y cierre con {accion_pago}.

    El LLM suele escribir su propia instrucción de pago ("actualizá tu tarjeta") después
    de la oferta, lo que CONTRADICE {accion_pago} (que depende del código de rechazo).
    Por eso cortamos todo lo que venga después de la frase de la oferta concreta y
    pegamos {accion_pago} como cierre. Determinístico: no dependemos de que el LLM obedezca.
    """
    t = (template or "").strip()
    if "{oferta_personal}" not in t:
        t = (t + " Tenemos algo para vos: {oferta_personal}.").strip()

    # Cortar al final de la oración que contiene {oferta_personal} (primer . ! ? posterior).
    idx = t.index("{oferta_personal}") + len("{oferta_personal}")
    fin = min((p for p in (t.find(c, idx) for c in ".!?") if p != -1), default=-1)
    cuerpo = (t if fin == -1 else t[:fin + 1]).strip()

    return f"{cuerpo} {{accion_pago}}"


def personalize_offer(
    offer: dict[str, Any],
    user: dict[str, Any],
    perfil_cluster: dict | None = None,
) -> str:
    """Rellena el template de la oferta con los datos reales del usuario.

    Combina tres fuentes deterministas (el LLM no toca ninguna):
      - {nombre} / {autor_favorito}: datos del usuario.
      - {oferta_personal}: oferta CONCRETA del catálogo (offer_matcher) según el incentivo
        del cluster y el historial del usuario — anti-alucinación.
      - {accion_pago}: qué hacer según el código de rechazo (recovery_actions).

    Acepta el dict de oferta completo (necesita incentivo_id) o, por compatibilidad, un
    string de template suelto.
    """
    if isinstance(offer, str):
        template = offer
        incentivo_id = "acceso_anticipado"
    else:
        template = offer.get("mensaje_template", "")
        incentivo_id = offer.get("incentivo_id", "acceso_anticipado")

    template = _asegurar_slots(template)

    # Oferta concreta desde el catálogo (libro/evento real, filtrado).
    rec = recomendar_oferta_personal(user, incentivo_id, perfil_cluster)
    # Acción de pago según el código de rechazo del usuario.
    accion = resolver_accion(_clean(user.get("decline_code")) or "_default")

    # Cláusula opcional que nombra al autor favorito (ej: "para vos, que seguís a X").
    # Solo se incluye si el item realmente ofrecido ES de ese autor; si la oferta cayó al
    # fallback por género (otro autor del mismo género), se omite para no sonar incoherente.
    fav_autor = _clean(user.get("autor_favorito"))
    oferta_autor = _clean((rec.get("item") or {}).get("autor")) or _clean(rec.get("autor"))
    autor_coincide = bool(fav_autor) and fav_autor.lower() == oferta_autor.lower()
    clausula_tpl = offer.get("clausula_autor", "") if isinstance(offer, dict) else ""
    clausula_autor = clausula_tpl.replace("{autor_favorito}", fav_autor) if (autor_coincide and clausula_tpl) else ""

    slots = _SafeDict({
        "nombre": _clean(user.get("first_name")) or "¡Hola!",
        "autor_favorito": fav_autor or "tus autores favoritos",
        "clausula_autor": clausula_autor,
        "oferta_personal": rec["texto"],
        "accion_pago": accion["texto"],
    })
    try:
        return template.format_map(slots)
    except Exception:
        return template


# ---------------------------------------------------------------------------
# Generación masiva + cache
# ---------------------------------------------------------------------------

def generate_all_offers(force: bool = False) -> dict[str, Any]:
    """Genera ofertas para todos los clusters (con perfil real) y cachea a disco."""
    if OFFERS_CACHE_PATH.exists() and not force:
        print(f"Cache ya existe en {OFFERS_CACHE_PATH}. Usá --force para regenerar.")
        with open(OFFERS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    # Capa A: perfiles reales desde datos (si hay CSV).
    perfiles: dict[str, Any] = {}
    try:
        from cluster_profiler import load_users, assign_clusters, build_cluster_profiles
        df = load_users()
        if df is not None:
            perfiles = build_cluster_profiles(assign_clusters(df))
            print("Perfiles de cluster cargados desde datos reales.")
        else:
            print("No hay CSV de usuarios; las ofertas usarán solo noticias.")
    except Exception as e:
        print(f"No se pudieron construir perfiles ({e}); sigo solo con noticias.")

    clusters = _load_clusters()
    cache: dict[str, Any] = {}

    print(f"\nGenerando ofertas para {len(clusters)} clusters...\n")
    for cluster_id, cluster_data in clusters.items():
        print(f"  [{cluster_id}] {cluster_data['nombre']}...")
        result = generate_offer(cluster_id, perfiles.get(cluster_id))
        cache[cluster_id] = result
        eco = result["economia"]
        flag = " (ajustado por guardrail)" if result.get("incentivo_ajustado") else ""
        print(f"    OK - incentivo: {eco['incentivo_nombre']}{flag} | "
              f"costo ARS {eco['costo_incentivo_ars']:.0f} / LTV ARS {eco['ltv_recuperado_ars']:.0f}")

    OFFERS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OFFERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\nCache guardado en {OFFERS_CACHE_PATH}")
    return cache


def load_cached_offer(cluster_id: str) -> dict[str, Any] | None:
    """Carga la oferta pregenerada de un cluster. None si no existe."""
    if not OFFERS_CACHE_PATH.exists():
        return None
    try:
        with open(OFFERS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f).get(cluster_id)
    except Exception:
        return None


def load_all_cached_offers() -> dict[str, Any]:
    """Carga todas las ofertas del cache. Dict vacío si no existe."""
    if not OFFERS_CACHE_PATH.exists():
        return {}
    try:
        with open(OFFERS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Composer de ofertas por cluster.")
    parser.add_argument("--force", action="store_true", help="Regenera aunque el cache exista.")
    parser.add_argument("--cluster", type=str, default=None, help="Genera solo para un cluster.")
    args = parser.parse_args()

    if args.cluster:
        from cluster_profiler import load_users, assign_clusters, build_cluster_profiles
        df = load_users()
        perfil = None
        if df is not None:
            perfil = build_cluster_profiles(assign_clusters(df)).get(args.cluster)
        result = generate_offer(args.cluster, perfil)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        offers = generate_all_offers(force=args.force)
        print("\n--- Resumen ---")
        for cid, data in offers.items():
            eco = data["economia"]
            print(f"\n[{cid}]  gancho: {data['gancho']}  | incentivo: {eco['incentivo_nombre']}")
            print(f"  template: {data['mensaje_template'][:140]}")
            print(f"  economia: costo ARS {eco['costo_incentivo_ars']:.0f} / "
                  f"margen ARS {eco['margen_por_recuperado_ars']:.0f} / "
                  f"guardrail {'OK' if eco['dentro_guardrail'] else 'AJUSTADO'}")
