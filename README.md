# Retenelo

**Recuperación de churn involuntario para suscripciones argentinas.**

> "Tu sistema de cobro es el barco. Retenelo es el salvavidas que recupera lo que el cobro no pudo cobrar."

Retenelo es una capa B2B que se activa **después** de que el sistema de billing del
cliente ya falló un cobro. Por cada pago caído, decide **a quién** recuperar, **qué**
ofrecerle, **cuándo** reintentar y **cómo** decírselo — y cobra comisión (10-15%) solo
sobre lo efectivamente recuperado. Costo inicial cero.

Todo evento entrante asume `attempt_number >= 1`: nunca somos el primer intento de cobro.

---

## La idea en una línea

> **El código decide QUÉ hacer y maneja los números. La IA solo decide CÓMO decirlo.
> Por diseño, no puede alucinar.**

La economía (costo del incentivo, LTV, margen) y la oferta concreta (qué título, qué
evento, qué fecha) las elige el **código** desde inventario verificado. El LLM solo
redacta el lenguaje del mensaje. Esto hace imposible que prometa algo que no existe.

---

## Cómo funciona (pipeline)

Por cada pago fallido, el evento recorre tres preguntas:

```
        Pago caído (decline code + datos del usuario)
                          │
   ┌──────────────────────┼──────────────────────┐
   ▼                      ▼                       ▼
¿QUIÉN ES?            ¿QUÉ HACEMOS?          ¿CÓMO SE LO DECIMOS?
clustering por       código de rechazo       el código arma los
comportamiento    →  + timing (calendario  → datos duros; la IA
(uso, frecuencia,    de sueldos) + un        solo elige las
recencia)            incentivo rentable      palabras
   │                      │                       │
   ▼                      ▼                       ▼
Capa A               Capa B                  Capa C
cluster_profiler     recovery_actions        offer_generator (LLM)
                     offer_generator         offer_matcher
                     (economía/guardrail)    (oferta concreta)
```

### Capa A — ¿Quién es? (perfilado)
`cluster_profiler.py` agrupa a los usuarios por comportamiento (cuánto consumen, con
qué frecuencia, hace cuánto que no entran) y calcula intereses dominantes y ARPU por
cluster. Determinístico, sin LLM.

### Capa B — ¿Qué hacemos? (decisión + economía)
- `recovery_actions.py` traduce el código de rechazo a la acción correcta:
  reintento automático (51 sin fondos, 65 límite → tras el día de cobro/resumen),
  actualizar tarjeta (54 vencida) o cargar otro medio (41 robada → nunca se reintenta).
- `offer_generator.py` (parte económica) elige un incentivo del catálogo y le adjunta
  el **costo real**, aplicando un *guardrail*: el incentivo nunca puede costar más de
  un % del LTV recuperado (`config.MARGEN_INCENTIVO_MAX`). Si lo supera, baja
  automáticamente a uno de costo marginal.
- `data/salary_calendar.py` aporta el timing argentino (quincena, fin de mes, ANSES,
  aguinaldo).

### Capa C — ¿Cómo se lo decimos? (redacción anti-alucinación)
- `offer_matcher.py` instancia una oferta **concreta** desde inventario verificado
  (`data/catalogo.json`), con una escalera de confianza individual → cluster → genérico.
  El núcleo anti-alucinación: el LLM nunca inventa un ítem.
- `offer_generator.py` (parte LLM) compone el mensaje rellenando slots
  (`{nombre}`, `{oferta_personal}`, `{accion_pago}`, …). Los números y la oferta ya
  vienen dados por el código.

---

## Estructura del proyecto

### Núcleo del demo (lo que corre en `app.py`)
| Archivo | Rol |
|---|---|
| `app.py` | Dashboard Streamlit (demo B2B). Punto de entrada. |
| `config.py` | Constantes, rutas, parámetros. Toda regla de negocio vive acá o en JSON. |
| `cluster_profiler.py` | Capa A — perfila clusters desde datos reales. |
| `recovery_actions.py` | Capa B — código de rechazo → acción de recuperación. |
| `offer_generator.py` | Capa B/C — economía del incentivo + composición LLM del mensaje. |
| `offer_matcher.py` | Capa C — oferta concreta desde inventario (anti-alucinación). |
| `simulator.py` | Calculadora de ROI para la vista B2B. |

### Datos
| Archivo | Contenido |
|---|---|
| `data/decline_codes.json` | Códigos Visa/MC/Payway + estrategia de recuperación. |
| `data/clusters.json` | Definición de los 5 segmentos de usuario. |
| `data/incentivos.json` | Catálogo de incentivos + tipo de costo. |
| `data/catalogo.json` | Inventario verificado (anti-alucinación). |
| `data/offers_cache.json` | Ofertas pregeneradas por cluster (para demo sin llamar al LLM). |
| `data/services/biblioteca.json` | Esquema de contexto del primer vertical. |
| `data/synthetic/dataset.csv` | Dataset sintético de usuarios con pagos fallidos. |
| `data/salary_calendar.py` | Calendario de sueldos argentino. |

### Generación de datos / entrenamiento (scripts standalone)
| Archivo | Rol |
|---|---|
| `generator.py` | Genera el dataset sintético con distribuciones argentinas reales. |
| `clustering.py` | Entrena el modelo KMeans de segmentación. |

### Módulos auxiliares / iteración (no en el camino del demo actual)
`model.py`, `recovery_engine.py`, `context_builder.py`, `output_composer.py`,
`salary_calendar.py` (shim), `crawler.py`, `domain_retriever.py` — versiones previas
del pipeline ML y la investigación de contexto/noticias. Se conservan para iteración.

---

## Setup

Requiere **Python 3.11+**.

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt
```

### API key del LLM (solo si regenerás ofertas)
El demo usa ofertas pregeneradas en `data/offers_cache.json`, así que **no necesita
API key para correr**. Para regenerar con el LLM, definí la variable de entorno:

```bash
set RETENELO_LLM_API_KEY=tu_api_key   # Windows
# export RETENELO_LLM_API_KEY=tu_api_key
```

---

## Uso

```bash
# Correr el dashboard (demo principal)
streamlit run app.py

# Regenerar el dataset sintético
python generator.py

# Entrenar el modelo de clustering
python clustering.py --train

# Regenerar las ofertas por cluster (requiere API key)
python offer_generator.py
```

El pitch deck está en `pitch.html` (abrir en el navegador).

---

## Cumplimiento regulatorio
- **Ley 25.326**: sin datos sensibles en los mensajes; opt-out fácil obligatorio.
- **BCRA / Payway**: tope de 15 reintentos por transacción rechazada en 30 días.
- **Defensa del Consumidor**: sin framing engañoso; nunca decimos "algo falló".

---

## Equipo
- **Jero** — Ciencia de Datos UBA
- **Gasti** — Ciencia de Datos UBA
- **Nico** — Ciencias de la Computacion UBA
