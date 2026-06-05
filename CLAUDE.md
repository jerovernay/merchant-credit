# MerchantCredit

## What we're building
A web app in Python + Streamlit that lets a merchant upload their sales reports
from Rappi, MercadoLibre and/or PedidosYa, computes an automatic credit score,
shows a pre-approved credit amount, and simulates repayment as a percentage of
future daily sales.

## Stack
- Python 3.11+
- Streamlit (UI)
- Pandas (data processing)
- Plotly (charts)
- Anthropic SDK with model claude-sonnet-4-6 (only to explain the score in natural language)
- API access is configured via AWS Bedrock (team credits)

## File structure
```
merchant-credit/
├── app.py                  # Streamlit entry point, 4-step flow
├── mock_api.py             # Realistic sample CSV generator
├── parser.py               # Multi-format parser per platform
├── scoring.py              # Deterministic underwriting engine
├── simulator.py            # Repayment and elasticity simulator
├── ai_explainer.py         # Score explanation via Claude
├── data/
│   ├── sample_rappi.csv
│   ├── sample_mercadopago.csv
│   └── sample_pedidosya.csv
├── requirements.txt
└── CLAUDE.md
```

## App flow (4 steps)
1. CONNECT — Upload one or more CSVs (Rappi, ML, PedidosYa) or use sample data
2. ANALYZE — Scoring engine computes variables and pre-approved amount
3. SIMULATE — Daily retention slider, repayment elasticity chart
4. APPLY — Acceptance form + visual digital signature flow (mock)

Each step only unlocks once the previous one is complete.

## CSV formats per platform

The parser must detect the platform by the columns present and normalize
everything into a unified DataFrame with columns: fecha, plataforma, monto_bruto,
monto_neto, tipo_operacion, estado.
(Keep these normalized column names in Spanish since the UI is in Spanish.)

### Mercado Pago / Mercado Libre
Columns: date_approved, payment_method_type, operation_type, purchase_value,
net_amount, sales_channel, merchant_id, pack_id, payer_id_type, payer_id_number,
payer_name
- operation_type can be: SETTLEMENT, REFUND, DISPUTE
- sales_channel can be: Delivery, Mercado_Pago, Mercado_Libre, Mercado_Shops

### PedidosYa
Columns: id_pedido, fecha, monto_bruto, comision_pedidosya, cobro_efectivo,
payout_net, saldo_acumulado
- Typical commission: 22%-27% + VAT

### Rappi
Columns: order_id, fecha, gross_value, rappi_commission, payout_status,
payment_method, rappi_score, cancellation_rate
- payout_status: PAID or PENDING
- payment_method: Online or Cash
- rappi_score: 1 to 5

## Scoring engine — exact formula

### The 5 variables to compute
1. V_neto — Consolidated net sales, monthly average (last 3 months)
2. CV — Daily Coefficient of Variation = std(daily_sales) / mean(daily_sales)
3. R_cancelaciones — Cancellation and incident rate (0 to 1 proportion)
4. T — Tenure: 1 if > 6 months of continuous activity, 0 otherwise
5. M_canal — Multi-channel: 1.2 if active on 2+ marketplaces, 1.0 if only one

### Maximum amount formula
```
Monto_Maximo = min(
    V_neto_mensual * (1.2 - CV) * (1 - R_cancelaciones) * M_canal * T,
    V_neto_mensual * 1.0  # business cap: max 1x monthly sales
)
```

### CV interpretation to show the user
- CV < 0.35 -> stable cash flow
- CV between 0.35 and 0.65 -> moderate cash flow
- CV > 0.65 -> unstable cash flow

### Scoring validations
- If R_cancelaciones > 0.01 (1%) -> risk flag, still compute but show a warning
- If T = 0 -> do not approve credit, show "insufficient tenure" message
- If Monto_Maximo < 800000 -> do not approve, minimum amount not reached
- Upper cap: Monto_Maximo cannot exceed 2,500,000 ARS

## MVP financial parameters
- Loan amount range: 800,000 to 2,500,000 ARS
- Factor Rate: 1.18 fixed for MVP (range 1.15–1.35 by risk tier at scale; merchant repays principal * 1.18 regardless of term)
- Daily retention: slider between 10% and 15% of sales (default 12%)
- Target term: 90 days

## Repayment simulator
Input: amount_disbursed, retention_pct, projected_daily_sales (DataFrame)
Output:
- Outstanding balance day by day (amount_disbursed * 1.18 - accumulated retention)
- Estimated full-repayment day
- Elasticity chart: how much is collected each day based on sales

Key rule: if a day's sales = 0, that day's retention = 0. No late fees, no penalties.

## Visualizations (all with Plotly)
1. "Pre-approved Limit" widget — large number in green with a button
2. Scoring radar chart — 5 axes with the 5 normalized variables
3. Retention slider wired to the estimated-term chart in real time
4. Repayment elasticity chart — decreasing balance line
5. Consolidated sales table by platform and month

## AI usage (claude-sonnet-4-6)
The AI is NOT used to compute the score (that is deterministic in scoring.py).
The AI is only used to:
- Generate a natural-language explanation of the obtained score
- Answer user questions about their profile ("why is my amount this?")
- Suggest 3 actions to improve the score

Process everything in a single API call per explanation to minimize cost.

## Code conventions
- UI strings, variable names that map to domain concepts, and comments: keep
  domain terms in Spanish (the product is for Argentine merchants), but you may
  write helper/internal code comments in English.
- Small functions with docstrings
- Read credentials from environment (configured via Bedrock); never hardcode keys
- No dependencies outside the defined stack
- Type hints on public functions
- Error handling: try/except with clear user-facing messages
- If the Anthropic API returns malformed JSON, parse with regex + fallback

## Verification
Before starting any task, briefly state how you will verify the output is correct.

## Commands
- Run: `streamlit run app.py`
- Install: `pip install -r requirements.txt`
