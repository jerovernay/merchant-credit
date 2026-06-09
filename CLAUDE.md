# Retenelo

## Verification
Before starting any task, briefly state how you will verify the output is correct.

## Project status (post-hackathon — building for real)
- Got 3rd place at the hackathon with an MVP. The goal now is a **real, sellable
  B2B product**. No rush — quality over speed.
- Core concept is confirmed; implementation is being rebuilt to production standards.
- Write modular code: business logic lives in config/JSON, never hardcoded inline.
- Output per service = **example messages**. There is no more Streamlit demo app.
- Full roadmap and current priorities live in `resume.txt` (repo root). Read it before
  proposing direction changes.

---

## What we're building
**Retenelo** — B2B SaaS that recovers involuntary churn for Argentine subscription
businesses. Commission 10-15% only on recovered revenue. Zero upfront cost.
Retenelo is the **rescue layer**: it activates ONLY after the client's billing system
has already failed. Every incoming event assumes `attempt_number >= 1`.

**Core pitch one-liner:**
"Rebill is your billing system. Retenelo is the insurance that recovers what your
billing system couldn't collect."

**Anti-hallucination principle (non-negotiable):**
The CODE decides WHAT to do and owns all numbers (incentive cost, LTV, margin) and
the concrete offer (which title/event/date, from verified inventory). The LLM only
decides HOW to say it. By design it cannot invent a price or an offer.

---

## Privacy by design (Ley 25.326 — a hard constraint, not a feature)
We must be able to operate **without ever receiving PII**.
- We NEVER receive the client's real name or exact location.
- We DO receive: the user's own usage history, age, and neighborhood/zone.
- No sensitive data ever appears in an outgoing message; easy opt-out is mandatory.
- The synthetic generator (`tools/generator.py`) still fabricates PII (name, DNI,
  address) inherited from the hackathon — that MUST be stripped as we go real.

---

## General, service-aware data model
The ingestion schema must be **general but service-aware**: a streaming service is not
a delivery service is not a book subscription. Each vertical defines its own behavioral
signals in `data/services/<vertical>.json`. Context — both macro (Argentine economy) and
per-service — lives in config/JSON, never inline. Vertical 1 (book subscription,
`biblioteca.json`) is the reference template; future verticals are decided as a group.

---

## Core pipeline (per failed payment event)
```
[1] Client data (PII-free: own history, age, neighborhood)
        ↓
[2] User typing → assigns user_type   (ML: clustering.py / model.py; rule-based twin exists)
        ↓
[3] Context enrichment (service + Argentine macro + geo tier) — from JSON, not hardcoded
        ↓
[4] Decline code classification (soft/hard, recoverable?, best_action)
        ↓
[5] Action triple: retry_window + channel + tone   (timing from salary_calendar)
        ↓
[6] Concrete incentive from verified inventory (anti-hallucination)
        ↓
[7] Message composition (claude-sonnet-4-6) — output = the message(s)
```

---

## Repo structure (packages — run with `python -m` from project root)
```
retenelo/
├── config.py                # ROOT. All paths + business constants. Top-level import.
├── resume.txt               # Roadmap + priorities (source of truth for direction)
├── pipeline/                # the real per-event path
│   ├── clustering.py          ML user typing (KMeans today; XGBoost target)
│   ├── model.py               XGBoost scoring / action selection
│   ├── recovery_engine.py     decline code → action triple (ML path)
│   ├── recovery_actions.py    decline code → action (rule-based twin, to be retired)
│   ├── cluster_profiler.py    rule-based cluster profiles (twin, to be retired)
│   ├── context_builder.py     context enrichment (service + macro + geo)
│   ├── output_composer.py     LLM message composer
│   └── salary_calendar.py     shim → data/salary_calendar.py
├── incentives/              # CORE — incentive selection + offer composition
│   ├── offer_generator.py     economics + LLM offer composition
│   └── offer_matcher.py       concrete offer from verified inventory
├── tools/                   # standalone scripts (not in the runtime path)
│   ├── generator.py           synthetic dataset (must drop PII)
│   ├── crawler.py             domain crawl (KEPT, to be built later)
│   └── domain_retriever.py    embedding retrieval (paired with crawler)
├── sales/
│   └── simulator.py           ROI calculator for the B2B pitch
├── data/                    # stays at root; config.py centralizes all paths
│   ├── decline_codes.json     Visa/MC/Payway codes + recovery strategy
│   ├── services/biblioteca.json   per-vertical schema (Vertical 1)
│   ├── catalogo.json          verified inventory (anti-hallucination)
│   ├── incentivos.json        incentive catalog + cost type
│   ├── clusters.json · geo_tiers.json · archetypes.json
│   ├── salary_calendar.py     Argentine payday logic
│   ├── crawler/ · synthetic/
├── models/                  # trained .pkl artifacts
├── docs/                    # pitch.html, clustering_spec.md
└── archive/                 # app.py (old Streamlit demo — not maintained)
```
Run examples: `python -m pipeline.clustering --train`, `python -m tools.generator`,
`python -m incentives.offer_generator`.

**Two pipelines coexist on purpose (for now).** Long-term direction is to consolidate
toward the ML path (`clustering.py`/`model.py` + `recovery_engine.py`) and retire the
rule-based twins (`cluster_profiler.py`, `recovery_actions.py`). Not done yet — the
incentive pipeline still depends on the rule-based ones. This is Phase 3/4 work.

---

## Decline codes
Full table in `data/decline_codes.json`. Key Argentine rules:
- **Code 51** — dominant (~50%). Retry after payday window.
- **Code 65** — over limit. Retry after statement date.
- **Code 54** — expired card. Requires user action.
- **Code 41/43** — lost/stolen. NEVER retry. Request alternate method.
- **Payway cap**: max 15 retry attempts per transaction within 30 days. Hard limit.
- Cabal débito: skip retries, go straight to user-action flow.

---

## Argentine salary calendar (`data/salary_calendar.py`)
- Quincena: 15th of month
- Fin de mes: last business day
- ANSES: by DNI-ending calendar
- Aguinaldo: June 30 and Dec 18-23 (liquidity spike — front-load retries)

---

## Synthetic data distribution (until real client data exists)
Synthetic data MUST reflect real Argentine distributions. Never uniform sampling.
Use `numpy.random.choice(options, p=weights)` everywhere.
- Code 51 must be ~50% of all failures.
- Weights live in service JSON configs, not hardcoded in `generator.py`.
- Generated profiles must be PII-free (history/age/neighborhood only).
- Replace synthetic data with real, anonymized client data as onboarding lands.

---

## LLM usage (claude-sonnet-4-6)
The ML pipeline decides WHAT to do; Claude decides HOW to communicate it.
- One API call per recovery event. Haiku for drafts/bulk, Sonnet for final output.
- `output_composer.py` is format-agnostic: returns `{"channel", "content", "metadata"}`.
- Never expose ML signals, scoring, or sensitive data to the end user.
- Always include the opt-out mention (Ley 25.326).
- The hackathon LiteLLM proxy is no longer the target; production LLM access is TBD.
  Never hardcode API keys (use env vars, e.g. `RETENELO_LLM_API_KEY`).

---

## Send channels (future — not built yet)
The product must send and **validate delivery** over WhatsApp, Email, and SMS. Today we
only produce the message text; the delivery layer (BSP integration, delivered/read
tracking, retry-cap enforcement, timestamped logs) is upcoming work (roadmap Phase 5).

---

## Roadmap (phased — see resume.txt for detail)
0. Hygiene + reorg — **done**
1. Data contract + privacy (general PII-free schema)
2. Company onboarding (self-serve DB connect — the Rappi ask)
3. Real ML (XGBoost; consolidate the typing/scoring modules)
4. Recovery engine + message (one composer, anti-hallucination)
5. Send channels (WhatsApp/Email/SMS + delivery validation)
6. Attribution + billing (which recovery was ours → commission)

---

## Competitive positioning
| Competitor | Gap | Our edge |
|------------|-----|----------|
| Rebill (YC+Tiger) | Bundled billing, charges regardless of recovery | Pay-on-recovery only; layers on top |
| Debi | ~32% recovery, no context enrichment | User typing + service context |
| Stripe/Paddle | No Argentine acquiring, no ARS, no WhatsApp | Local-first |
| In-house scripts | Naive retries, no compliance | ML timing + 15-attempt cap enforced |

Rebill claims 71% recovery — unaudited. Real audited benchmarks: 32-50%.
We promise less and measure everything.

---

## Regulatory compliance
- Ley 25.326: no sensitive data in output, easy opt-out mandatory, no PII ingested.
- BCRA/Payway: max 15 retry attempts per declined transaction within 30 days.
- Defensa del Consumidor: no deceptive framing; never say "something failed."
- Log all retry attempts with timestamps.

---

## Team split
- **Jero**: core pipeline + end-user interaction flow (this repo).
- **Gasti**: research on alternative recovery methods beyond direct messaging.
- **Nico**: B2B dashboard — what the client company sees.

Keep modules decoupled. Expose clean function returns so Nico's layer can consume them
without touching the core pipeline.

---

## Code conventions
- UI strings and domain terms: Spanish. Internal comments: English.
- Type hints on all public functions; docstrings required.
- Config over hardcoding — everything in JSON/config.
- try/except with Spanish user-facing error messages.
- Never hardcode API keys.

---

## Commands
- Install: `pip install -r requirements.txt`
- Generate synthetic data: `python -m tools.generator`
- Train clustering: `python -m pipeline.clustering --train`
- Regenerate cluster offers (needs API key): `python -m incentives.offer_generator`
