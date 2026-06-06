# Esquema del dataset sintético (`dataset.csv`)

Fuente de verdad de las columnas que produce `generator.py` y que consume `model.py`.
Las etiquetas (`recovered`, `recovery_day_offset`), los identificadores (`id_evento`,
`user_id`, `fecha_evento`) y la textura (`contexto_cliente`) **no** son features del modelo.

| Grupo | Columna | Tipo | Descripción |
|-------|---------|------|-------------|
| id/meta | `id_evento` | str | ID único del evento de pago fallido |
| id/meta | `fecha_evento` | date (ISO) | Fecha del rechazo |
| payment | `decline_code` | str | Código ISO-8583 (ver `decline_codes.json`) |
| payment | `decline_tipo` | str | `soft` \| `hard` |
| payment | `red` | str | Visa / Mastercard / Cabal / Amex / Naranja |
| payment | `acquirer` | str | Adquirente (Payway / Fiserv / Geopagos / Other) |
| payment | `bin` | str | Primeros 6 dígitos (prefijo por red) |
| payment | `card_type` | str | `credito` \| `debito` |
| payment | `monto_ars` | float | Monto del cargo fallido (ARS) |
| payment | `attempt_number` | int | Nº de intento (>= 1; Retenelo entra tras retries nativos) |
| payment | `tokenized` | bool | Tarjeta tokenizada |
| payment | `card_expiry_delta_dias` | int | Días hasta vencimiento (negativo = vencida) |
| user | `user_id` | str | ID del cliente |
| user | `archetype_id` | str | Arquetipo (ver `archetypes.json`) |
| user | `employment_type` | str | Tipo de empleo (del arquetipo) |
| user | `digital_literacy` | str | alta / media / baja |
| user | `tenure_meses` | int | Antigüedad como suscriptor |
| user | `prior_recoveries` | int | Recuperaciones previas |
| user | `engagement_recency_dias` | int | Días desde la última interacción |
| user | `arpu_ars` | float | Ingreso medio mensual por usuario (ARS) |
| timing | `day_of_month` | int | Día del mes del evento |
| timing | `days_to_quincena` | int | Días a la próxima quincena |
| timing | `days_to_fin_de_mes` | int | Días al último día hábil del mes |
| timing | `is_aguinaldo_month` | int | 1 si junio/diciembre |
| timing | `anses_pay_flag` | int | 1 si cae en ventana aprox. de pago ANSES `[PENDING]` |
| external | `indec_ipc_mom` | float | Inflación mensual (proxy) `[PENDING fuente real]` |
| external | `bcra_rate` | float | Tasa BCRA anual (proxy) `[PENDING]` |
| external | `billetera_yield_proxy` | float | Rendimiento de billetera virtual (proxy) `[PENDING]` |
| action | `action_window` | str | Ventana de reintento registrada (ver `config.VENTANAS_REINTENTO`) |
| action | `action_channel` | str | Canal registrado (ver `config.CANALES`) |
| action | `action_tone` | str | Tono registrado (ver `config.TONOS`) |
| **label** | `recovered` | int (0/1) | Target principal: se recuperó el pago |
| **label** | `recovery_day_offset` | int | Target secundario: días hasta recuperar (-1 si no) |
| textura | `contexto_cliente` | str | Nota libre (Haiku o plantilla numpy). **No es feature.** |

## Notas de diseño
- **Distribuciones no uniformes:** arquetipos y códigos se muestrean con
  `prevalencia_weight` (numpy `rng.choice(p=weights)`). Código 51 mayoritario.
- **Etiqueta no determinista:** `recovered ~ Bernoulli(p)` con `p` derivada de
  arquetipo × código × timing × canal × intentos + ruido gaussiano (evita circularidad).
- **Reproducibilidad:** todo el backbone depende de `--seed`.
