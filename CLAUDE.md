# Retenelo

## Verification
Before starting any task, briefly state how you will verify the output is correct.

## Project status (Hackathon — live)
- Project name and core concept are **confirmed**.
- Architecture and ML details are **subject to change** as research evolves.
- Write modular code: keep business logic (decline-code rules, archetype definitions,
  salary-calendar windows) in config files or dedicated modules — never hardcoded inline.
- When in doubt, build the simplest thing that works and expose the right hooks for iteration.

---

## What we're building
**Retenelo** — a B2B SaaS platform that recovers involuntary churn for Argentine
subscription businesses. We charge 10-15% commission **only on recovered revenue**
(zero upfront cost). The client keeps their existing billing stack; Retenelo layers on top.

Involuntary churn = payment failures due to technical reasons (card declined, insufficient
funds, expired card, fraud flags) — NOT voluntary cancellations.

**Core pitch one-liner:**
"Rebill is your billing system. Retenelo is the insurance that recovers what your billing
system couldn't collect."

---

## Stack
- Python 3.11+
- Streamlit (UI / demo dashboard)
- Pandas + scikit-learn / XGBoost (ML pipeline)
- Plotly (charts)
- Anthropic SDK — model `claude-sonnet-4-6` (message personalization only)
- API via hackathon proxy (LiteLLM over AWS Bedrock):
  - Base URL: `https://litellm-alb-1708856422.us-east-1.elb.amazonaws.com`
  - Models: `claude-sonnet-4-6`, `claude-haiku-4-5`
  - Budget: USD 195 (shared; use Haiku for data generation, Sonnet for final output)
  - TLS: set `NODE_TLS_REJECT_UNAUTHORIZED=0` / `verify=False` in SDK calls

---

## File structure
```
retenelo/
├── app.py                  # Streamlit entry point — demo dashboard
├── data/
│   ├── archetypes.json     # 23 user archetypes (from research)
│   ├── decline_codes.json  # Visa/MC/Payway codes with recovery strategy
│   ├── salary_calendar.py  # Argentine payday logic (quincena, fin de mes, ANSES, aguinaldo)
│   └── synthetic/          # Generated training CSVs (via Haiku)
├── generator.py            # Synthetic dataset generator (uses claude-haiku-4-5)
├── model.py                # XGBoost scorer + contextual bandit action layer
├── recovery_engine.py      # Core pipeline: decline code -> action triple
├── message_composer.py     # Claude Sonnet: personalized message per archetype + channel
├── simulator.py            # ROI calculator for pitch demo
├── requirements.txt
└── CLAUDE.md
```

---

## App flow (demo)
1. **INPUT** — Enter or simulate a failed payment (decline code + user archetype)
2. **SCORE** — ML model outputs: retry window + contact channel + message tone
3. **COMPOSE** — Claude Sonnet generates the personalized recovery message
4. **SIMULATE** — ROI calculator: how much the client recovers vs. Retenelo's commission

---

## Core domain: decline codes

Codes split into **soft** (retriable) and **hard** (require user action or stop).
Full table in `data/decline_codes.json`. Key Argentine rules:

- **Code 51 (Insufficient Funds)** — dominant in Argentina. Retry AFTER payday windows.
- **Code 65 (Over limit)** — retry after statement payment date (~1-10 of month).
- **Code 54 (Expired card)** — requires user action (update card or Account Updater).
- **Code 41/43 (Lost/Stolen)** — NEVER retry. Request alternate method.
- **Payway rule:** max 15 retry attempts per declined transaction within 30 days.
  Exceeding triggers fines. Compliance is mandatory, not optional.
- **Cabal debito** — effectively zero collectability for debito automatico. Skip retries,
  go straight to user-action dunning.
- **Argentine-specific:** "Autorizacion denegada por el emisor" = most common local reason.
  During 2023 Payway tokenization change, 70%+ of Visa Credito rejections carried this.
  Channel rotation across merchant numbers recovered ~98% of those cases.

---

## Core domain: user archetypes

23 archetypes defined in `data/archetypes.json`. Key fields per archetype:
`name, age_band, employment_type, financial_behavior, likely_decline_code,
best_channel, best_retry_window, digital_literacy`

Strategic insight: **Code 51 dominates Argentina** (amplified by salary-cycle liquidity
gaps and pesos parked in remunerated FCI accounts like Mercado Pago / Naranja X that
don't auto-fund card debits). **Timing the retry to the salary calendar is the
single highest-leverage action.**

Channel defaults:
- WhatsApp -> universal best channel for most archetypes
- Phone call -> elderly / low-digital archetypes only
- Email (formal) -> corporate / executive profiles

---

## Core domain: Argentine salary calendar
Key windows in `data/salary_calendar.py`:
- **Quincena**: 15th of each month
- **Fin de mes**: last business day of month
- **ANSES**: by DNI-ending calendar (public sector / retirees)
- **Aguinaldo**: June 30 (+4 business days grace) and Dec 18-23
  -> Front-load retry intensity in these windows for code-51 failures.

---

## ML model design

### What it predicts (multi-output)
Per failed payment, the optimal **(retry timing window, contact channel, message tone)**
triple that maximizes recovery probability while respecting network retry caps.

### Architecture
1. **XGBoost** — core recovery-probability scorer (tabular data, handles mixed features)
2. **Contextual bandit** — action selection layer (retry timing x channel); epsilon-greedy
   exploration for new clients with sparse history
3. **Multi-output heads** — jointly predict timing + channel + tone

### Key features
- Payment: decline code, Visa/MAC category, acquirer, BIN, card type, amount ARS,
  prior attempt count, tokenized flag, card-expiry delta
- User: tenure, prior recovery history, engagement recency, ARPU, archetype label
- Timing: day-of-month, days-to-quincena, days-to-fin-de-mes, is-aguinaldo-month,
  ANSES pay calendar
- External: INDEC IPC (inflation proxy), BCRA rate, billetera-yield proxy

### Synthetic dataset
Generated via `claude-haiku-4-5` in `generator.py`.
Full column schema in `data/synthetic/schema.md`.
Use Haiku for generation to preserve budget; Sonnet only for final message output.

### Dataset distribution (critical)
The synthetic dataset MUST reflect real-world Argentine distributions, not uniform sampling.
Define explicit probability weights in `generator.py` for both archetypes and decline codes.
Rare archetypes (e.g. posgrado becario, petrolero) must be minorities (~2-3% each).
Common archetypes (estudiante, monotributista, empleado publico) must dominate.
Code 51 must be the majority decline code, matching Argentine market reality.
Use `numpy.random.choice(options, p=weights)` to enforce this on every generated row.
Weights are research-informed estimates — update them if mentor feedback or real data
provides better calibration. Never use uniform distribution as a default.

---

## Message personalization (claude-sonnet-4-6)

Claude is used **only** to compose the outbound recovery message.
The ML model decides WHAT action to take; Claude decides HOW to phrase it.

Per call, pass:
- Archetype profile
- Decline code + human-readable reason
- Chosen channel (WhatsApp / email / SMS / push)
- Tone recommendation from model
- Argentine regulatory constraints (Ley 25.326)

Output: a single recovery message ready to send, in Spanish.
One API call per recovery event. Use Haiku for bulk drafting, Sonnet for final.

---

## ROI simulator (pitch demo)

Inputs: subscriber count, ARPU (ARS), monthly failure rate %, Retenelo recovery %,
commission %, average post-recovery retention months.

Reference benchmark (mid-market Argentine SaaS):
- 1,000 subs x ARS 5,000 ARPU -> ARS 5M MRR
- 8% failure rate -> ARS 400K at risk monthly
- 40% recovery -> ARS 160K recovered
- 12.5% commission -> ARS 20K to Retenelo; ARS 140K net to client
- Annual net benefit: ARS 1.68M on zero upfront cost

---

## Competitive positioning

| Competitor | What they do | Our edge |
|------------|-------------|----------|
| Rebill (YC + Tiger) | Full billing platform, retries bundled | Pay-on-recovery only; plug in on top of Rebill |
| Debi | Recovery specialist, ~32% recovery | Deeper decline-code intelligence + WhatsApp-first |
| Stripe/Paddle | Global, Stripe-dependent | No Argentine acquiring; no ARS; no WhatsApp |
| In-house scripts | Naive daily retries | ML timing + compliance guardrails (15-attempt cap) |

**Key counter to Rebill:** Their 71% recovery claim is unaudited marketing.
Audited real-world data shows 32-50%. We promise less and measure everything.

---

## Regulatory compliance (non-negotiable)

- **Ley 25.326** (data protection): no sensitive data in messages, easy opt-out
- **BCRA**: max 15 retry attempts per declined transaction within 30 days (Payway rule)
- **Defensa del Consumidor**: no deceptive framing in recovery messages
- **WhatsApp Business API**: only utility templates for transactional recovery messages
- Log all retry attempts with timestamps for compliance audit trail

---

## Code conventions
- UI strings and domain terms: Spanish (product is for Argentine merchants)
- Internal code comments: English
- Small functions with docstrings and type hints on public functions
- Config over hardcoding: decline-code strategies, archetype definitions, and
  salary-calendar rules live in JSON/config, not in business logic
- Error handling: try/except with clear user-facing messages in Spanish
- Never hardcode API keys; read from environment
- TLS: `verify=False` / `NODE_TLS_REJECT_UNAUTHORIZED=0` for hackathon proxy

---

## Commands
- Run: `streamlit run app.py`
- Install: `pip install -r requirements.txt`
- Generate synthetic data: `python generator.py`
- Train model: `python model.py --train`
