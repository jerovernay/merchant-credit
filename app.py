# === app.py ===
"""Dashboard de recuperación de churn involuntario — Retenelo.

Uso:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from cluster_profiler import assign_clusters, build_cluster_profiles, load_users
from offer_generator import (
    generate_offer,
    load_all_cached_offers,
    load_cached_offer,
    personalize_offer,
)
from simulator import calculate_roi

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Retenelo — Recuperación de Suscriptores",
    page_icon="🔄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_clusters() -> dict:
    with open(config.CLUSTERS_PATH, encoding="utf-8") as f:
        return json.load(f)["clusters"]


@st.cache_data
def load_users_df() -> pd.DataFrame | None:
    df = load_users()
    return assign_clusters(df) if df is not None else None


@st.cache_data
def load_profiles(_df: pd.DataFrame) -> dict:
    return build_cluster_profiles(_df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_ars(value: float) -> str:
    return f"ARS {value:,.0f}".replace(",", ".")


def get_offer(cluster_id: str, perfil: dict | None) -> dict:
    """Carga la oferta del cache o la genera en vivo si no existe."""
    cached = load_cached_offer(cluster_id)
    if cached is not None:
        return cached
    with st.spinner("Generando oferta con IA..."):
        try:
            return generate_offer(cluster_id, perfil)
        except Exception:
            return {
                "tipo_gancho": "noticia",
                "gancho": "—",
                "incentivo_id": "acceso_anticipado",
                "mensaje_template": (
                    "Hola {nombre}, no queremos que te pierdas tu próxima lectura. "
                    "Tenemos algo para vos: {oferta_personal}. {accion_pago}"
                ),
                "justificacion": "Respaldo genérico.",
                "economia": {"incentivo_nombre": "Acceso anticipado", "costo_incentivo_ars": 0,
                             "ltv_recuperado_ars": 0, "margen_por_recuperado_ars": 0,
                             "dentro_guardrail": True},
                "incentivo_ajustado": False,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    clusters = load_clusters()
    df = load_users_df()

    if df is not None:
        counts = df["cluster_id"].value_counts().to_dict()
        total_usuarios = len(df)
        profiles = load_profiles(df)
    else:
        counts = {cid: 0 for cid in clusters}
        total_usuarios = 0
        profiles = {}

    # --- Header ---
    st.title("🔄 Retenelo")
    st.caption("Recuperación de churn involuntario · Vertical: Biblioteca de suscripción")
    st.divider()

    # --- Sidebar: selector + métricas globales ---
    with st.sidebar:
        st.header("Clusters de usuarios")

        cluster_ids = list(clusters.keys())
        cluster_options = {
            cid: f"{clusters[cid]['icono']}  {clusters[cid]['nombre']}  ({counts.get(cid, 0)})"
            for cid in cluster_ids
        }
        selected_id = st.radio(
            "Seleccioná un segmento:",
            options=cluster_ids,
            format_func=lambda cid: cluster_options[cid],
            key="cluster_selector",
        )

        st.divider()
        st.subheader("Resumen global")
        st.metric("Usuarios con pago fallido", total_usuarios or "—")
        st.metric("Clusters activos", len(clusters))

        if Path("data/offers_cache.json").exists():
            st.success("Ofertas pregeneradas ✓")
        else:
            st.warning("Cache no encontrado.\nCorré: `python offer_generator.py`")

    # --- Cluster seleccionado ---
    cluster = clusters[selected_id]
    perfil = profiles.get(selected_id)
    cluster_users = df[df["cluster_id"] == selected_id] if df is not None else pd.DataFrame()
    n_cluster = len(cluster_users)

    # --- Métricas del cluster ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segmento", f"{cluster['icono']} {cluster['nombre']}")
    col2.metric("Usuarios afectados", n_cluster or "—")
    col3.metric("Libros / mes (prom.)", perfil["libros_mes_prom"] if perfil else cluster["caracteristicas"]["libros_mes"])
    col4.metric("ARPU estimado", fmt_ars(perfil["arpu_ars"]) if perfil else "—")

    st.caption(f"_{cluster['descripcion']}_")

    # --- Intereses reales del segmento (Capa A) ---
    if perfil:
        ci1, ci2 = st.columns(2)
        with ci1:
            autores = " · ".join(f"{a['valor']} ({a['n']})" for a in perfil["top_autores"]) or "—"
            st.markdown(f"**✍️ Autores más leídos:** {autores}")
        with ci2:
            generos = " · ".join(f"{g['valor']} ({g['n']})" for g in perfil["top_generos"]) or "—"
            st.markdown(f"**📖 Géneros más leídos:** {generos}")

    st.divider()

    # --- Oferta ---
    offer = get_offer(selected_id, perfil)
    eco = offer["economia"]

    st.subheader("💡 Oferta de recuperación")
    badge = {"autor": "🎯 Anclada en autor", "genero": "📚 Anclada en género",
             "noticia": "📰 Anclada en noticia"}.get(offer.get("tipo_gancho"), "")
    st.caption(f"{badge} · Gancho: *{offer.get('gancho', '—')}* · Tono: *{cluster['tono_oferta']}*")

    col_eco, col_news = st.columns([3, 2], gap="large")

    with col_eco:
        # Economía del incentivo — el código es dueño de los números
        st.markdown(f"**Incentivo elegido:** {eco['incentivo_nombre']}")
        e1, e2, e3 = st.columns(3)
        e1.metric("Costo / recuperado", fmt_ars(eco["costo_incentivo_ars"]))
        e2.metric("LTV recuperado", fmt_ars(eco["ltv_recuperado_ars"]))
        e3.metric("Margen neto", fmt_ars(eco["margen_por_recuperado_ars"]))

        if offer.get("incentivo_ajustado"):
            st.warning("⚖️ El incentivo original superaba el guardrail de margen; "
                       "se bajó automáticamente a un incentivo de costo marginal.")
        else:
            ratio = eco.get("ratio_costo_ltv", 0)
            st.success(f"✅ Dentro del guardrail: el incentivo cuesta {ratio:.0%} del LTV "
                       f"(tope {config.MARGEN_INCENTIVO_MAX:.0%}).")

        if offer.get("justificacion"):
            st.caption(f"_Razón: {offer['justificacion']}_")

        st.markdown("**Template del mensaje** _(se rellena por usuario)_:")
        st.code(offer["mensaje_template"], language=None)

    with col_news:
        st.markdown("**📰 Contexto de noticias**")
        for noticia in cluster["noticias_relevantes"][:3]:
            with st.container(border=True):
                st.markdown(f"**{noticia['titulo']}**")
                st.caption(f"Fuente: {noticia['fuente']}")

    # --- Previews personalizados por usuario ---
    st.divider()
    st.subheader("✉️ Mensajes individualizados (preview)")
    st.caption("El mismo template, rellenado por usuario: oferta concreta del catálogo "
               "(no inventada) + acción según su código de rechazo.")

    from offer_matcher import recomendar_oferta_personal
    from recovery_actions import resolver_accion

    conf_badge = {"individual": "🎯 dato propio", "cluster": "👥 prior del segmento",
                  "generico": "🌱 cold-start"}
    accion_badge = {"reintento_auto": "🔄 reintento automático",
                    "actualizar_datos": "💳 actualizar tarjeta",
                    "metodo_alternativo": "🆕 otro medio de pago"}

    if not cluster_users.empty:
        for _, user in cluster_users.head(3).iterrows():
            u = user.to_dict()
            mensaje = personalize_offer(offer, u, perfil)
            rec = recomendar_oferta_personal(u, offer["incentivo_id"], perfil)
            accion = resolver_accion(str(u.get("decline_code") or "_default"))
            autor = str(u.get("autor_favorito") or "").strip()
            autor = autor if autor.lower() not in ("", "nan") else "—"

            st.info(mensaje, icon="✉️")
            st.caption(
                f"→ {u.get('first_name', '')} {u.get('last_name', '')} · "
                f"{u.get('ciudad', '')} · autor fav: {autor} · "
                f"oferta: {conf_badge.get(rec['confianza'], rec['confianza'])} "
                f"({rec['tipo_recomendacion']}) · "
                f"rechazo {u.get('decline_code', '—')} ({accion['descripcion']}): "
                f"{accion_badge.get(accion['tipo'], accion['tipo'])}"
            )
    else:
        st.info("Generá el dataset primero: `python generator.py`")

    # --- Envío simulado ---
    st.divider()
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        enviar = st.button(
            f"📨 Simular envío a {n_cluster or '—'} usuarios",
            type="primary",
            disabled=(n_cluster == 0),
            key=f"enviar_{selected_id}",
        )
    with col_info:
        st.caption(
            "En producción el envío es batch + async por el canal óptimo de cada usuario "
            "(WhatsApp / email / push), con dedupe y tope de reintentos Payway."
        )

    if enviar:
        with st.spinner(f"Enviando ofertas a {n_cluster} usuarios..."):
            import time; time.sleep(1.5)
        st.success(
            f"✅ {n_cluster} mensajes individualizados enviados al segmento "
            f"**{cluster['nombre']}**. Costo total de incentivos estimado: "
            f"{fmt_ars(eco['costo_incentivo_ars'] * n_cluster * config.RECOVERY_RATE_BASE)}."
        )
        st.balloons()

    # --- Muestra de usuarios ---
    st.divider()
    with st.expander(f"👥 Ver datos crudos del segmento — {cluster['nombre']}"):
        if not cluster_users.empty:
            cols = [
                "first_name", "last_name", "ciudad", "genero_favorito",
                "autor_favorito", "libros_leidos_ultimos_3_meses",
                "frecuencia_apertura_app", "ultimo_acceso_dias", "decline_code",
            ]
            cols = [c for c in cols if c in cluster_users.columns]
            sample = cluster_users[cols].head(10).rename(columns={
                "first_name": "Nombre", "last_name": "Apellido", "ciudad": "Ciudad",
                "genero_favorito": "Género", "autor_favorito": "Autor favorito",
                "libros_leidos_ultimos_3_meses": "Libros (3m)",
                "frecuencia_apertura_app": "Frecuencia",
                "ultimo_acceso_dias": "Días sin acceso", "decline_code": "Código rechazo",
            })
            st.dataframe(sample, use_container_width=True, hide_index=True)
        else:
            st.info("Generá el dataset primero: `python generator.py`")

    # --- ROI Calculator ---
    st.divider()
    st.subheader("📊 Simulador de ROI")
    st.caption("Estimá el impacto económico de la recuperación para tu empresa")

    r1, r2, r3 = st.columns(3)
    n_fallidos = r1.number_input("Pagos fallidos por mes", min_value=1, value=total_usuarios or 80, step=10)
    ticket = r2.number_input("Ticket promedio (ARS)", min_value=100,
                             value=int(perfil["arpu_ars"]) if perfil else 7000, step=500)
    tasa = r3.slider("Tasa de recuperación estimada", min_value=0.10, max_value=0.70,
                     value=0.40, step=0.05, format="%.0f%%")

    roi = calculate_roi(n_failed_payments=n_fallidos, avg_ticket_ars=ticket, recovery_rate=tasa)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monto en riesgo", fmt_ars(roi["monto_en_riesgo_ars"]))
    m2.metric("Revenue recuperado", fmt_ars(roi["revenue_recuperado_ars"]))
    m3.metric("Comisión Retenelo", fmt_ars(roi["comision_retenelo_ars"]))
    m4.metric("Ingreso neto cliente", fmt_ars(roi["ingreso_neto_cliente_ars"]), delta=f"ROI {roi['roi_pct']:.1f}%")


if __name__ == "__main__":
    main()
