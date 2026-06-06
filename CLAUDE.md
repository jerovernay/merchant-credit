# Retenelo

## Verification
Before starting any task, briefly state how you will verify the output is correct.

## Project status (Hackathon — live)
- Core concept confirmed. Many implementation details subject to change.
- Write modular code: business logic lives in config/JSON, never hardcoded inline.
- Build the simplest thing that works and expose clean hooks for iteration.
- Output format per service is TBD — keep it decoupled from the pipeline.

---

## What we're building
**Retenelo** — B2B SaaS that recovers involuntary churn for Argentine subscription
businesses. Commission 10-15% only on recovered revenue. Zero upfront cost.
Retenelo is the rescue layer: it activates ONLY after the client's billing system
has already failed. Every incoming event assumes attempt_number >= 1.

**Core pitch one-liner:**
"Rebill is your billing system. Retenelo is the insurance that recovers what your
billing system couldn't collect."

---

## Stack
- Python 3.11+
- Streamlit (demo dashboard)
- Pandas + scikit-learn / XGBoost (ML pipeline)
- Plotly (charts)
- Anthropic SDK — model `claude-sonnet-4-6` (output personalization only)
- API via hackathon proxy (LiteLLM over AWS Bedrock):
  - Base URL: `https://litellm-alb-1708856422.us-east-1.elb.amazonaws.com`
  - Models: `claude-sonnet-4-6`, `claude-haiku-4-5`
  - Budget: USD 195 shared. Haiku for data generation, Sonnet for final output only.
  - TLS: `verify=False` / `NODE_TLS_REJECT_UNAUTHORIZED=0`

---

## Core pipeline (per failed payment event)

```
[1] Raw user data (from client company)
        ↓
[2] User clustering model → assigns user_type (e.g. "lector_voraz", "casual")
        ↓
[3] Context enrichment:
    - Service context (books read, ratings, favorite genre/author, usage frequency)
    - Economic context (Argentine macro: inflation, salary calendar, BCRA rate)
    - Geographic context (province/city tier)
        ↓
[4] Decline code classification (soft/hard, recoverable?, best_action)
        ↓
[5] Action triple: retry_window + channel + tone
        ↓
[6] Output generation (claude-sonnet-4-6) — format TBD per service
```

---

## File structure
```
retenelo/
├── app.py                   # Streamlit demo dashboard
├── config.py                # Global constants, paths, model params
├── data/
│   ├── decline_codes.json   # Visa/MC/Payway codes + recovery strategy
│   ├── services/
│   │   └── biblioteca.json  # Service-specific context schema (first vertical)
│   └── synthetic/           # Generated training data
├── generator.py             # Synthetic dataset generator (Haiku)
├── clustering.py            # User type clustering (KMeans or similar)
├── context_builder.py       # Enriches user profile with service + macro + geo context
├── recovery_engine.py       # Decline code -> action triple
├── salary_calendar.py       # Argentine payday logic
├── output_composer.py       # Claude Sonnet: generates personalized output
├── simulator.py             # ROI calculator
├── requirements.txt
└── CLAUDE.md
```

---

## Service verticals (3 total — only #1 defined so far)

### Vertical 1: Biblioteca de suscripcion (reference: Bukku, Escape a Pluton)
Target: monthly book subscription platforms.

**Data from signup form:**
- first_name, last_name, country, street_address, ciudad, state, postcode,
  phone, email, cuit_dni, payment_email (MercadoPago)

**Data from in-app behavior:**
- libros_leidos_total, libros_leidos_ultimos_3_meses
- rating_promedio_dado, autor_favorito, genero_favorito
- frecuencia_apertura_app (daily/weekly/monthly)
- ultimo_acceso_dias (days since last login)
- lista_deseos_activa (bool), resenas_escritas

**User types to cluster (examples — calibrate with data):**
- lector_voraz: high frequency, many books, varied genres
- lector_fiel: moderate frequency, specific genre/author loyalty
- lector_casual: low frequency, few books read
- coleccionista: low reading, high wishlist activity
- inactivo_reciente: was active, dropped off last 30-60 days

### Vertical 2: [PENDING — to be defined]
### Vertical 3: [PENDING — to be defined]

---

## Decline codes
Full table in `data/decline_codes.json`. Key Argentine rules:
- **Code 51** — dominant (~50%). Retry after payday window.
- **Code 65** — over limit. Retry after statement date.
- **Code 54** — expired card. Requires user action.
- **Code 41/43** — lost/stolen. NEVER retry. Request alternate method.
- **Payway cap**: max 15 retry attempts per transaction within 30 days. Hard limit.
- Cabal debito: skip retries, go straight to user-action flow.

---

## Argentine salary calendar (`salary_calendar.py`)
- Quincena: 15th of month
- Fin de mes: last business day
- ANSES: by DNI-ending calendar
- Aguinaldo: June 30 and Dec 18-23 (liquidity spike — front-load retries)

---

## Dataset distribution (critical)
Synthetic data MUST reflect real Argentine distributions. Never uniform sampling.
Use `numpy.random.choice(options, p=weights)` everywhere.
- Code 51 must be ~50% of all failures
- User type weights must match realistic subscription demographics
- Weights live in service JSON configs, not hardcoded in generator.py
- Update weights if mentor feedback or real data provides better calibration

---

## Output generation (claude-sonnet-4-6)
Claude composes the final output per recovery event.
The ML pipeline decides WHAT to do; Claude decides HOW to communicate it.
Output format (message, push, email, etc.) is TBD per vertical — keep
output_composer.py format-agnostic: it returns a dict with `channel` and `content`.
One API call per recovery event. Haiku for drafts/bulk, Sonnet for final only.

**Per call, pass:**
- user_type + enriched context (service + macro + geo)
- decline_code + human-readable reason
- action triple (retry_window, channel, tone)
- Ley 25.326 compliance flag (no sensitive data, easy opt-out)

---

## Competitive positioning
| Competitor | Gap | Our edge |
|------------|-----|----------|
| Rebill (YC+Tiger) | Bundled billing, charges regardless of recovery | Pay-on-recovery only; layers on top |
| Debi | ~32% recovery, no context enrichment | User clustering + service context |
| Stripe/Paddle | No Argentine acquiring, no ARS, no WhatsApp | Local-first |
| In-house scripts | Naive retries, no compliance | ML timing + 15-attempt cap enforced |

Rebill claims 71% recovery — unaudited. Real audited benchmarks: 32-50%.
We promise less and measure everything.

---

## Regulatory compliance
- Ley 25.326: no sensitive data in output, easy opt-out mandatory
- BCRA/Payway: max 15 retry attempts per declined transaction within 30 days
- Defensa del Consumidor: no deceptive framing
- Log all retry attempts with timestamps

---

## Team split
- **Jero**: core pipeline + end-user interaction flow (this repo)
- **Gasti**: research on alternative recovery methods beyond direct messaging
- **Nico**: B2B dashboard — what the client company sees

Keep modules decoupled. Expose clean function returns so Nico's layer can consume
them without touching the core pipeline.

---

## Code conventions
- UI strings and domain terms: Spanish
- Internal comments: English
- Type hints on all public functions, docstrings required
- Config over hardcoding — everything in JSON/config
- try/except with Spanish user-facing error messages
- Never hardcode API keys

---

## Commands
- Run: `streamlit run app.py`
- Install: `pip install -r requirements.txt`
- Generate data: `python generator.py`
- Train: `python clustering.py --train`
