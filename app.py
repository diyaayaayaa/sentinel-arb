"""
Sentinel-Arb: Geopolitical Stress-Tester & Basis Monitor
=========================================================
Tornado backend — now with live market data integration.
"""

import os
import sys
import json
import asyncio
import time
import logging
from typing import Optional

# ── Load .env ────────────────────────────────────────────────────────────
# Try multiple locations: same dir as script, and current working directory
for _env_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(os.getcwd(), ".env"),
]:
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()
        print(f"Loaded .env from: {_env_path}")
        break

import tornado.ioloop
import tornado.web
import numpy as np

from simulation.market_data import PalmOilMarket
from simulation.shock_engine import (
    get_logistics_shock,
    get_policy_shock,
    build_custom_shock,
    list_scenarios,
)
from simulation.ai_analysis import interpret_headline, generate_traders_brief
from simulation.live_data import get_fetcher
from simulation.news_feed import NewsFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("sentinel-arb")

# ── Global instances ─────────────────────────────────────────────────────
market = PalmOilMarket()
fetcher = get_fetcher()
news_feed = NewsFeed(market=market, ai_scorer=interpret_headline)


# ── Base handler ─────────────────────────────────────────────────────────
class JsonHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def write_json(self, data):
        self.write(json.dumps(data))

    def get_json_body(self):
        try:
            body = self.request.body
            if not body:
                return {}
            return json.loads(body.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError, Exception):
            return {}

    def write_error(self, status_code, **kwargs):
        self.set_header("Content-Type", "application/json")
        import traceback
        exc = kwargs.get("exc_info")
        msg = traceback.format_exception(*exc)[-1].strip() if exc else "Internal server error"
        self.write(json.dumps({"error": msg, "status": status_code}))

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()


# ── Market Data ──────────────────────────────────────────────────────────
class MarketCurrentHandler(JsonHandler):
    def get(self):
        if market.history:
            self.write_json(market.history[-1])
        else:
            self.write_json(market.tick())


class MarketHistoryHandler(JsonHandler):
    def get(self):
        n = int(self.get_argument("n", "90"))
        n = max(1, min(500, n))
        self.write_json(market.get_history(n))


class MarketSpreadHandler(JsonHandler):
    def get(self):
        latest = market.history[-1] if market.history else market.tick()
        base = latest["base_spread"]
        spread = latest["spread"]
        self.write_json({
            "spread": spread,
            "base_spread": base,
            "deviation": round(spread - base, 2),
            "deviation_pct": round(((spread - base) / base) * 100, 2) if base else 0,
            "shock_active": latest["shock_active"],
        })


class DataStatusHandler(JsonHandler):
    """Returns the current data-source status (live vs simulated)."""
    def get(self):
        live = fetcher.get_prices()
        self.write_json({
            "is_live": live.get("is_live", False),
            "source": live.get("source", "simulated"),
            "last_updated": live.get("last_updated"),
            "market_status": live.get("market_status", "unknown"),
            "error": live.get("error"),
            "fob_live": live.get("fob_malaysia"),
            "cif_live": live.get("cif_rotterdam"),
        })


# ── Shock Simulation ────────────────────────────────────────────────────
class ScenariosHandler(JsonHandler):
    def get(self):
        self.write_json(list_scenarios())


class LogisticsShockHandler(JsonHandler):
    async def post(self, scenario_key):
        shock = get_logistics_shock(scenario_key)
        if not shock:
            self.set_status(404)
            self.write_json({"error": f"Unknown logistics scenario: {scenario_key}"})
            return
        market.inject_shock(shock)
        latest = market.history[-1] if market.history else market.tick()
        combined = market.get_combined_shock_state()
        brief = await generate_traders_brief(
            shock, latest["spread"], latest["fob_malaysia"], latest["cif_rotterdam"],
            combined_state=combined,
        )
        self.write_json({"status": "shock_injected", "shock": shock, "traders_brief": brief, "combined_state": combined})


class PolicyShockHandler(JsonHandler):
    async def post(self, scenario_key):
        shock = get_policy_shock(scenario_key)
        if not shock:
            self.set_status(404)
            self.write_json({"error": f"Unknown policy scenario: {scenario_key}"})
            return
        market.inject_shock(shock)
        latest = market.history[-1] if market.history else market.tick()
        combined = market.get_combined_shock_state()
        brief = await generate_traders_brief(
            shock, latest["spread"], latest["fob_malaysia"], latest["cif_rotterdam"],
            combined_state=combined,
        )
        self.write_json({"status": "shock_injected", "shock": shock, "traders_brief": brief, "combined_state": combined})


class HeadlineShockHandler(JsonHandler):
    async def post(self):
        body = self.get_json_body()
        headline = body.get("headline", "").strip()
        if not headline or len(headline) < 5:
            self.set_status(400)
            self.write_json({"error": "Headline must be at least 5 characters"})
            return

        analysis = await interpret_headline(headline)
        shock = build_custom_shock(
            freight_impact=analysis["freight_impact"],
            spread_multiplier=analysis["multiplier"],
            duration_hours=analysis["duration_estimate_hours"],
            description=f"AI-interpreted: {headline}",
            severity=analysis["severity"],
        )
        market.inject_shock(shock)
        latest = market.history[-1] if market.history else market.tick()
        combined = market.get_combined_shock_state()
        brief = await generate_traders_brief(
            shock, latest["spread"], latest["fob_malaysia"], latest["cif_rotterdam"],
            combined_state=combined,
        )
        self.write_json({
            "status": "ai_shock_injected",
            "analysis": analysis,
            "shock": shock,
            "traders_brief": brief,
            "combined_state": combined,
        })


class CustomShockHandler(JsonHandler):
    async def post(self):
        body = self.get_json_body()
        try:
            freight_impact    = float(body.get("freight_impact", 0))
            spread_multiplier = float(body.get("spread_multiplier", 1.0))
            duration_hours    = float(body.get("duration_hours", 24))
        except (ValueError, TypeError) as e:
            self.set_status(400)
            self.write_json({"error": f"Invalid numeric field: {e}"})
            return
        # Clamp to safe ranges: multiplier must be positive, duration must be > 0
        spread_multiplier = max(0.01, min(10.0, spread_multiplier))
        duration_hours    = max(0.1,  min(8760.0, duration_hours))  # 0.1h – 1 year
        shock = build_custom_shock(
            freight_impact=freight_impact,
            spread_multiplier=spread_multiplier,
            duration_hours=duration_hours,
            description=body.get("description", "Custom shock event"),
        )
        market.inject_shock(shock)
        latest = market.history[-1] if market.history else market.tick()
        combined = market.get_combined_shock_state()
        brief = await generate_traders_brief(
            shock, latest["spread"], latest["fob_malaysia"], latest["cif_rotterdam"],
            combined_state=combined,
        )
        self.write_json({"status": "custom_shock_injected", "shock": shock, "traders_brief": brief, "combined_state": combined})


class ClearShocksHandler(JsonHandler):
    def post(self):
        market.clear_shocks()
        self.write_json({"status": "all_shocks_cleared"})


class ActiveShocksHandler(JsonHandler):
    def get(self):
        now = time.time()
        active = []
        # Use get_combined_shock_state (thread-safe) to get current active shocks
        combined = market.get_combined_shock_state()
        for s in combined.get("shocks", []):
            shock_copy = {k: v for k, v in s.items() if k != "expires"}
            shock_copy["remaining_seconds"] = max(0, round(s.get("expires", now) - now))
            active.append(shock_copy)
        self.write_json({"count": len(active), "shocks": active})


# ── P&L ──────────────────────────────────────────────────────────────────
class PnLHandler(JsonHandler):
    def post(self):
        body = self.get_json_body()
        entry = float(body.get("entry_spread", 0))
        exit_spread = body.get("exit_spread")
        volume = float(body.get("volume_mt", 5000))
        cost = float(body.get("transaction_cost", 2500))
        if exit_spread is None:
            latest = market.history[-1] if market.history else market.tick()
            exit_spread = latest["spread"]
        else:
            exit_spread = float(exit_spread)
        self.write_json(market.calculate_pnl(entry, exit_spread, volume, cost))


# ── Geopolitical Events ─────────────────────────────────────────────────
class MainPageHandler(tornado.web.RequestHandler):
    def initialize(self, static_path):
        self.static_path = static_path

    def get(self):
        self.set_header("Content-Type", "text/html")
        with open(os.path.join(self.static_path, "index.html")) as f:
            self.write(f.read())


# ── News Feed ──────────────────────────────────────────────────────────
class NewsFeedHandler(JsonHandler):
    """Return the latest scored news headlines."""
    def get(self):
        limit = int(self.get_argument("limit", "20"))
        self.write_json({
            "headlines": news_feed.get_feed(limit),
            "stats": news_feed.get_stats(),
        })


class NewsPollHandler(JsonHandler):
    """Force an immediate poll cycle (for manual refresh)."""
    async def post(self):
        result = await news_feed.poll_and_score()
        self.write_json(result)


class NewsAutoInjectHandler(JsonHandler):
    """Toggle auto-injection on/off."""
    def post(self):
        body = self.get_json_body()
        enabled = body.get("enabled", True)
        news_feed.auto_inject_enabled = bool(enabled)
        self.write_json({
            "auto_inject_enabled": news_feed.auto_inject_enabled,
        })


class NewsInjectHandler(JsonHandler):
    """Manually inject a specific scored headline as a shock."""
    async def post(self, item_id):
        item = next((i for i in news_feed.items if i.id == item_id), None)
        if not item:
            self.set_status(404)
            self.write_json({"error": f"Headline {item_id} not found"})
            return
        if not item.scored:
            self.set_status(400)
            self.write_json({"error": "Headline has not been scored yet"})
            return

        from simulation.shock_engine import build_custom_shock
        shock = build_custom_shock(
            freight_impact=item.freight_impact,
            spread_multiplier=item.multiplier,
            duration_hours=item.duration_hours,
            description=f"[NEWS] {item.title}",
            severity=item.severity,
        )
        market.inject_shock(shock)
        item.auto_injected = True

        latest = market.history[-1] if market.history else market.tick()
        combined = market.get_combined_shock_state()
        brief = await generate_traders_brief(
            shock, latest["spread"], latest["fob_malaysia"], latest["cif_rotterdam"],
            combined_state=combined,
        )
        self.write_json({
            "status": "news_shock_injected",
            "headline": item.title,
            "shock": shock,
            "traders_brief": brief,
            "combined_state": combined,
        })


class GeoEventsHandler(JsonHandler):
    def get(self):
        self.write_json([
            {
                "id": 1, "type": "hotspot", "region": "Malacca Strait",
                "severity": "high",
                "title": "Naval Exercise Disrupts Shipping Lane",
                "summary": "Unscheduled military exercise near Malacca Strait narrows causes vessels to reduce speed. Minor delays reported.",
                "lat": 2.5, "lon": 101.5, "timestamp": "2026-04-02T08:30:00Z",
            },
            {
                "id": 2, "type": "sanctions", "region": "Indonesia",
                "severity": "medium",
                "title": "Indonesia Mulls CPO Export Levy Increase",
                "summary": "Indonesian trade ministry considering raising CPO export levy by $15/MT to fund biodiesel subsidy program.",
                "lat": -2.5, "lon": 118.0, "timestamp": "2026-04-02T06:15:00Z",
            },
            {
                "id": 3, "type": "weather", "region": "South China Sea",
                "severity": "medium",
                "title": "Tropical Depression Forming Near Philippines",
                "summary": "Tropical depression may strengthen to typhoon within 48 hours. Could disrupt shipping routes between Malaysia and East Asia.",
                "lat": 12.0, "lon": 125.0, "timestamp": "2026-04-02T04:00:00Z",
            },
            {
                "id": 4, "type": "conflict", "region": "Red Sea",
                "severity": "high",
                "title": "Vessel Attacked Near Bab el-Mandeb",
                "summary": "Bulk carrier hit by drone near Bab el-Mandeb strait. War risk premiums for Red Sea transit increasing. Some operators rerouting via Cape of Good Hope.",
                "lat": 12.5, "lon": 43.3, "timestamp": "2026-04-01T22:00:00Z",
            },
            {
                "id": 5, "type": "outage", "region": "Rotterdam",
                "severity": "low",
                "title": "Rotterdam Refinery Maintenance Extended",
                "summary": "Major palm oil refinery in Rotterdam extends maintenance shutdown by 5 days. Local refined palm oil stocks tighten.",
                "lat": 51.9, "lon": 4.5, "timestamp": "2026-04-01T14:30:00Z",
            },
        ])


# ── Health Check ────────────────────────────────────────────────────────
class HealthHandler(JsonHandler):
    """Simple health check endpoint for monitoring."""
    def get(self):
        latest = market.history[-1] if market.history else {}
        self.write_json({
            "status": "ok",
            "uptime_ticks": len(market.history),
            "active_shocks": len(market.active_shocks),
            "data_mode": latest.get("data_mode", "simulated"),
            "news_tracked": len(news_feed.items),
            "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


# ── Background tickers ──────────────────────────────────────────────────
async def market_ticker():
    """Advance the market every 3 seconds. Restarts automatically on error."""
    while True:
        try:
            market.tick()
        except Exception as e:
            logger.error("market_ticker error: %s", e)
        await asyncio.sleep(3)


async def live_anchor_loop():
    """Re-anchor to live prices every 30 seconds. Restarts on error."""
    while True:
        try:
            prices = fetcher.get_prices()
            if prices.get("is_live"):
                market.anchor_to_live(prices)
        except Exception as e:
            logger.error("live_anchor_loop error: %s", e)
        await asyncio.sleep(30)


async def news_feed_loop():
    """Poll RSS feeds, score with Haiku, auto-inject HIGH-confidence shocks."""
    await asyncio.sleep(8)  # Let market initialize first
    while True:
        try:
            await news_feed.poll_and_score()
        except Exception as e:
            logger.error("news_feed_loop error: %s", e)
        await asyncio.sleep(news_feed.fetch_interval)


# ── Application ──────────────────────────────────────────────────────────
def make_app():
    static_path = os.path.join(os.path.dirname(__file__), "static")
    return tornado.web.Application([
        # Market data
        (r"/api/market/current", MarketCurrentHandler),
        (r"/api/market/history", MarketHistoryHandler),
        (r"/api/market/spread", MarketSpreadHandler),
        (r"/api/data-status", DataStatusHandler),
        # Scenarios
        (r"/api/scenarios", ScenariosHandler),
        # Shocks
        (r"/api/shock/logistics/(.*)", LogisticsShockHandler),
        (r"/api/shock/policy/(.*)", PolicyShockHandler),
        (r"/api/shock/headline", HeadlineShockHandler),
        (r"/api/shock/custom", CustomShockHandler),
        (r"/api/shock/clear", ClearShocksHandler),
        (r"/api/shock/active", ActiveShocksHandler),
        # P&L
        (r"/api/pnl/calculate", PnLHandler),
        # News feed
        (r"/api/news/feed", NewsFeedHandler),
        (r"/api/news/poll", NewsPollHandler),
        (r"/api/news/auto-inject", NewsAutoInjectHandler),
        (r"/api/news/inject/(.*)", NewsInjectHandler),
        # Health
        (r"/api/health", HealthHandler),
        # Geo events
        (r"/api/geo/events", GeoEventsHandler),
        # Static + frontend
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path}),
        (r"/", MainPageHandler, {"static_path": static_path}),
    ], debug=False)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app = make_app()
    app.listen(port)

    # Start the live price fetcher background thread
    fetcher.start()

    # ── Startup diagnostics ───────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    ai_status = f"✓ Configured ({api_key[:12]}...)" if api_key else "✗ NOT FOUND — AI will use simulation"

    print()
    print("=" * 62)
    print("  SENTINEL-ARB  Geopolitical Stress-Tester & Basis Monitor")
    print("=" * 62)
    print(f"  Dashboard    :  http://localhost:{port}")
    print(f"  Claude Haiku :  {ai_status}")
    print(f"  Data mode    :  Live fetch + simulation fallback")
    print(f"  News feed    :  RSS → Haiku scoring → auto-inject (every 3 min)")
    print("=" * 62)
    print("  Ctrl+C to stop the server")
    print("=" * 62)
    print()

    loop = asyncio.get_event_loop()
    loop.create_task(market_ticker())
    loop.create_task(live_anchor_loop())
    loop.create_task(news_feed_loop())
    tornado.ioloop.IOLoop.current().start()
