# Sentinel-Arb 🛢️

**Geopolitical Stress-Tester & Palm Oil Basis Monitor**

A real-time trading simulation dashboard that models how geopolitical shocks, port blockages, export bans, weather events, affect the **CIF Rotterdam − FOB Malaysia palm oil basis spread**.

---

## Features

- **Live market simulation** using an Ornstein-Uhlenbeck mean-reverting stochastic process, anchored to real-world prices via the World Bank Commodity API
- **Geopolitical shock injection** — pre-configured logistics and policy scenarios (Malacca blockage, Suez disruption, Indonesia export ban, EU deforestation regulation, and more)
- **Compound shock awareness** — inject multiple simultaneous shocks; the engine correctly stacks freight impacts (additive) and spread multipliers (multiplicative)
- **Live RSS news feed** — fetches palm oil, shipping, and trade policy headlines every 10 minutes, scored by Claude Haiku for market impact
- **Auto-injection** — HIGH-confidence headlines (severity: high/critical + multiplier ≥ 1.15x) are automatically injected as shocks
- **AI Trader's Brief** — Claude Haiku generates a quantitative risk brief for every shock, reasoning over the full compounded state of all active shocks
- **Apple-inspired dashboard** — earthy/pastel palette, ticker tape, smooth transitions, scrollable layout

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Tornado async web framework |
| Market engine | NumPy, Ornstein-Uhlenbeck OU process |
| AI analysis | Anthropic Claude Haiku (via `anthropic` SDK) |
| Live data | World Bank Commodity API, Yahoo Finance, FRED (fallback chain) |
| News feed | Google News RSS → XML parsing → Haiku scoring |
| Frontend | Vanilla JS, Chart.js, CSS animations |

---

## Project Structure

```
sentinel-arb/
├── app.py                    # Tornado server + all API endpoints
├── simulation/
│   ├── market_data.py        # OU price engine, shock stacking, thread-safe state
│   ├── shock_engine.py       # Pre-configured logistics & policy scenarios
│   ├── ai_analysis.py        # Claude Haiku integration (headline scoring + Trader's Brief)
│   ├── news_feed.py          # RSS fetcher → scorer → auto-injector
│   └── live_data.py          # Live price fetcher (World Bank → Yahoo → FRED)
├── static/
│   └── index.html            # Full dashboard (single-file, no build step)
├── tests/
│   └── test_edge_cases.py    # 121-test edge case & stress test suite
├── .env.example              # API key template
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/sentinel-arb.git
cd sentinel-arb
pip3 install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Open .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Without an API key the dashboard runs in **simulation mode** — all market logic and shocks work, but AI analysis uses deterministic fallback responses instead of real Haiku output.

### 3. Run the server

```bash
python3 app.py
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/market/current` | Latest tick (FOB, CIF, spread, data mode) |
| GET | `/api/market/history?n=90` | Last N ticks |
| GET | `/api/market/spread` | Spread deviation from fair value |
| GET | `/api/data-status` | Live vs simulated data source |
| GET | `/api/scenarios` | All pre-configured shock scenarios |
| POST | `/api/shock/logistics/{key}` | Inject a logistics shock |
| POST | `/api/shock/policy/{key}` | Inject a policy shock |
| POST | `/api/shock/headline` | AI-interpret a custom headline |
| POST | `/api/shock/custom` | Inject a fully custom shock |
| POST | `/api/shock/clear` | Clear all active shocks |
| GET | `/api/shock/active` | List all currently active shocks |
| GET | `/api/news/feed` | Latest scored headlines |
| POST | `/api/news/poll` | Force an immediate RSS poll |
| POST | `/api/news/auto-inject` | Toggle auto-injection on/off |
| POST | `/api/news/inject/{id}` | Manually inject a scored headline |
| GET | `/api/health` | Server health check |

---

## Running Tests

```bash
python3 tests/test_edge_cases.py
```

121 tests covering:
- Market engine edge cases (clamps, NaN, thread safety)
- Shock stacking & compound state correctness
- AI analysis JSON extraction robustness
- News feed RSS parsing & memory cap
- API input validation & error handling
- Endurance / stress tests (1000 ticks, concurrent threads)

---

## Shock Scenarios

**Logistics**
- `malacca_blockage` — Malacca Strait blockage (1.25x spread, +$35/MT freight, 72h)
- `suez_blockage` — Suez Canal disruption (1.18x, +$25/MT, 96h)
- `port_klang_outage` — Port Klang outage (1.10x, +$15/MT, 48h)
- `rotterdam_congestion` — Rotterdam congestion (1.08x, +$10/MT, 120h)

**Policy**
- `indonesia_export_ban` — Indonesia export ban (1.20x, 168h)
- `b50_mandate` — Indonesia B50 biodiesel mandate (1.15x, 720h)
- `eu_deforestation_reg` — EUDR enforcement (1.12x, 720h)
- `india_import_duty` — India import duty hike (0.92x — spread narrows)

---

## Compound Shock Logic

When multiple shocks are active simultaneously:
- **Freight impacts are summed**: $35 + $25 = $60/MT total
- **Spread multipliers are compounded**: 1.25 × 1.18 = 1.475x total
- **Trader's Brief reasons over the full compound state**, not just the newest shock

---

*Built with Python, Tornado, and Claude Haiku.*
