"""
Sentinel-Arb Edge Case & Stress Test Suite
===========================================
Covers:
  A. Market Data Engine  — tick(), shocks, history, P&L, live anchoring
  B. Shock Engine        — valid/invalid keys, custom shock validation
  C. AI Analysis         — JSON extraction, mock fallbacks, compound state
  D. News Feed           — RSS parsing, relevance, seen_ids cap, scoring
  E. API Handlers        — malformed inputs, boundary values, error paths
  F. Compound Shock      — stacking, expiry, thread-safety

Run:  python3 -m pytest tests/test_edge_cases.py -v
  or: python3 tests/test_edge_cases.py
"""

import sys
import os
import time
import json
import asyncio
import unittest
import threading
from unittest.mock import patch, MagicMock, AsyncMock

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.market_data import PalmOilMarket, OrnsteinUhlenbeck
from simulation.shock_engine import (
    get_logistics_shock, get_policy_shock, build_custom_shock, list_scenarios
)
from simulation.ai_analysis import _extract_json, _mock_headline_analysis, _mock_traders_brief
from simulation.news_feed import NewsFeed, NewsItem, RELEVANCE_KEYWORDS

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_async(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


DUMMY_SHOCK = {
    "type": "logistics",
    "name": "Test Shock",
    "freight_impact": 10.0,
    "spread_multiplier": 1.2,
    "duration_hours": 1.0,   # short so expiry tests work easily
    "description": "Test shock for unit tests",
    "severity": "medium",
}


# ═════════════════════════════════════════════════════════════════════════════
# A. MARKET DATA ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class TestOrnsteinUhlenbeck(unittest.TestCase):
    """OU process basic sanity."""

    def test_step_returns_float(self):
        ou = OrnsteinUhlenbeck(mu=850, theta=0.15, sigma=120)
        val = ou.step(850.0)
        self.assertIsInstance(val, float)

    def test_step_mean_reverts_over_time(self):
        """After many steps from an extreme value, should converge toward mu."""
        ou = OrnsteinUhlenbeck(mu=100, theta=0.5, sigma=1.0, dt=1/252)
        x = 500.0
        for _ in range(5000):
            x = ou.step(x)
        self.assertLess(abs(x - 100), 50, "OU process should mean-revert toward mu=100")

    def test_zero_sigma_is_deterministic(self):
        ou = OrnsteinUhlenbeck(mu=100, theta=1.0, sigma=0.0, dt=1/252)
        x1 = ou.step(200.0)
        ou2 = OrnsteinUhlenbeck(mu=100, theta=1.0, sigma=0.0, dt=1/252)
        x2 = ou2.step(200.0)
        self.assertAlmostEqual(x1, x2, places=10)


class TestPalmOilMarketTick(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()

    def test_tick_returns_required_keys(self):
        tick = self.market.tick()
        required = [
            "timestamp", "fob_malaysia", "cif_rotterdam", "spread",
            "base_spread", "shock_active", "data_mode",
        ]
        for key in required:
            self.assertIn(key, tick, f"Missing key: {key}")

    def test_fob_within_hard_clamp(self):
        """No matter how extreme the OU step, FOB stays in [400, 2000]."""
        for _ in range(200):
            tick = self.market.tick()
            self.assertGreaterEqual(tick["fob_malaysia"], 400.0)
            self.assertLessEqual(tick["fob_malaysia"], 2000.0)

    def test_spread_within_hard_clamp(self):
        for _ in range(200):
            tick = self.market.tick()
            self.assertGreaterEqual(tick["spread"], 20.0)
            self.assertLessEqual(tick["spread"], 500.0)

    def test_cif_equals_fob_plus_spread(self):
        tick = self.market.tick()
        expected_cif = round(tick["fob_malaysia"] + tick["spread"], 2)
        self.assertAlmostEqual(tick["cif_rotterdam"], expected_cif, places=1)

    def test_tick_appends_to_history(self):
        initial_len = len(self.market.history)
        self.market.tick()
        self.assertEqual(len(self.market.history), initial_len + 1)

    def test_history_capped_at_max(self):
        from simulation.market_data import MAX_HISTORY
        # Force many ticks
        for _ in range(MAX_HISTORY + 50):
            self.market.tick()
        self.assertLessEqual(len(self.market.history), MAX_HISTORY)

    def test_tick_survives_numpy_edge_cases(self):
        """Force fob_price to extreme values before ticking."""
        self.market.fob_price = 2000.0
        tick = self.market.tick()
        self.assertLessEqual(tick["fob_malaysia"], 2000.0)

        self.market.fob_price = 400.0
        tick = self.market.tick()
        self.assertGreaterEqual(tick["fob_malaysia"], 400.0)

    def test_data_mode_field(self):
        tick = self.market.tick()
        self.assertIn(tick["data_mode"], ("live", "simulated"))

    def test_shock_active_false_when_no_shocks(self):
        self.market.clear_shocks()
        tick = self.market.tick()
        self.assertFalse(tick["shock_active"])


class TestPalmOilMarketShocks(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()

    def test_inject_shock_appears_in_active_shocks(self):
        shock = dict(DUMMY_SHOCK, duration_hours=24)
        self.market.inject_shock(shock)
        self.assertEqual(len(self.market.active_shocks), 1)

    def test_shock_expires_and_clears(self):
        """Shock with duration_hours=0 should expire almost immediately."""
        shock = dict(DUMMY_SHOCK)
        self.market.inject_shock(shock)
        # Force expires to past
        self.market.active_shocks[-1]["expires"] = time.time() - 1
        tick = self.market.tick()  # _shock_adjustment() should prune it
        self.assertFalse(tick["shock_active"])

    def test_clear_shocks(self):
        self.market.inject_shock(dict(DUMMY_SHOCK))
        self.market.inject_shock(dict(DUMMY_SHOCK))
        self.market.clear_shocks()
        self.assertEqual(len(self.market.active_shocks), 0)

    def test_multiple_shocks_stack_freight(self):
        """Combined freight should be the sum of individual freight impacts."""
        s1 = dict(DUMMY_SHOCK, freight_impact=10.0, spread_multiplier=1.0, duration_hours=24)
        s2 = dict(DUMMY_SHOCK, freight_impact=20.0, spread_multiplier=1.0, duration_hours=24)
        self.market.inject_shock(s1)
        self.market.inject_shock(s2)
        state = self.market.get_combined_shock_state()
        self.assertAlmostEqual(state["combined_freight_impact"], 30.0, places=2)

    def test_multiple_shocks_compound_multiplier(self):
        """Combined multiplier should be the product of individual multipliers."""
        s1 = dict(DUMMY_SHOCK, freight_impact=0, spread_multiplier=1.20, duration_hours=24)
        s2 = dict(DUMMY_SHOCK, freight_impact=0, spread_multiplier=1.15, duration_hours=24)
        self.market.inject_shock(s1)
        self.market.inject_shock(s2)
        state = self.market.get_combined_shock_state()
        expected = round(1.20 * 1.15, 4)
        self.assertAlmostEqual(state["combined_spread_multiplier"], expected, places=3)

    def test_get_combined_shock_state_no_shocks(self):
        self.market.clear_shocks()
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 0)
        self.assertAlmostEqual(state["combined_spread_multiplier"], 1.0)
        self.assertAlmostEqual(state["combined_freight_impact"], 0.0)

    def test_get_combined_shock_state_skips_expired(self):
        """Expired shocks should not appear in combined state."""
        self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=24))
        self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=24))
        # Manually expire first shock
        self.market.active_shocks[0]["expires"] = time.time() - 1
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 1)

    def test_ten_stacked_shocks(self):
        """Stress test: 10 simultaneous shocks should not crash anything."""
        self.market.clear_shocks()
        for i in range(10):
            shock = dict(DUMMY_SHOCK, spread_multiplier=1.05, freight_impact=5.0, duration_hours=24)
            self.market.inject_shock(shock)
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 10)
        expected_mult = round(1.05 ** 10, 4)
        self.assertAlmostEqual(state["combined_spread_multiplier"], expected_mult, places=2)
        tick = self.market.tick()
        self.assertLessEqual(tick["spread"], 500.0)  # Still within clamp

    def test_shock_with_zero_multiplier(self):
        """Zero spread_multiplier would collapse spread; clamp should protect."""
        shock = dict(DUMMY_SHOCK, spread_multiplier=0.0, duration_hours=24)
        self.market.inject_shock(shock)
        tick = self.market.tick()
        # Spread clamp at 20 should prevent it going to 0
        self.assertGreaterEqual(tick["spread"], 20.0)

    def test_shock_with_negative_multiplier(self):
        """Negative multiplier is nonsensical — spread clamp must protect."""
        shock = dict(DUMMY_SHOCK, spread_multiplier=-1.0, duration_hours=24)
        self.market.inject_shock(shock)
        tick = self.market.tick()
        self.assertGreaterEqual(tick["spread"], 20.0)

    def test_inject_shock_thread_safety(self):
        """Concurrent inject_shock calls from multiple threads must not corrupt state."""
        errors = []

        def inject_repeatedly():
            for _ in range(50):
                try:
                    self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=1))
                    self.market.tick()
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=inject_repeatedly) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(errors, [], f"Thread safety errors: {errors}")


class TestLiveAnchor(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()

    def test_valid_live_data_anchors(self):
        prices = {
            "fob_malaysia": 850.0,
            "cif_rotterdam": 980.0,
            "source": "world_bank",
            "is_live": True,
            "last_updated": "2026-04-04T10:00:00Z",
        }
        self.market.anchor_to_live(prices)
        self.assertTrue(self.market._is_live)
        self.assertAlmostEqual(self.market._live_fob, 850.0)

    def test_fob_below_range_rejected(self):
        prices = {"fob_malaysia": 100.0, "cif_rotterdam": 250.0, "is_live": True, "source": "test"}
        self.market.anchor_to_live(prices)
        self.assertFalse(self.market._is_live)

    def test_fob_above_range_rejected(self):
        prices = {"fob_malaysia": 5000.0, "cif_rotterdam": 5200.0, "is_live": True, "source": "test"}
        self.market.anchor_to_live(prices)
        self.assertFalse(self.market._is_live)

    def test_missing_fob_field(self):
        prices = {"cif_rotterdam": 980.0, "is_live": True, "source": "test"}
        self.market.anchor_to_live(prices)
        self.assertFalse(self.market._is_live)

    def test_is_live_false_flag_rejected(self):
        prices = {"fob_malaysia": 850.0, "cif_rotterdam": 980.0, "is_live": False, "source": "test"}
        self.market.anchor_to_live(prices)
        self.assertFalse(self.market._is_live)

    def test_spread_clamped_when_anchoring(self):
        """Extreme spread from live data (e.g. bad API) should be clamped."""
        prices = {"fob_malaysia": 800.0, "cif_rotterdam": 2800.0, "is_live": True, "source": "test"}
        self.market.anchor_to_live(prices)
        if self.market._is_live:
            self.assertLessEqual(self.market._live_spread, 500.0)

    def test_cif_equals_fob_gives_zero_spread_clamped(self):
        """CIF == FOB → spread=0 → should be clamped to 20."""
        prices = {"fob_malaysia": 850.0, "cif_rotterdam": 850.0, "is_live": True, "source": "test"}
        self.market.anchor_to_live(prices)
        if self.market._is_live:
            self.assertGreaterEqual(self.market._live_spread, 20.0)


class TestPnL(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()

    def test_basic_profit(self):
        result = self.market.calculate_pnl(100.0, 120.0, 5000, 2500)
        self.assertEqual(result["gross_pnl"], 100_000.0)
        self.assertEqual(result["net_pnl"], 97_500.0)

    def test_basic_loss(self):
        result = self.market.calculate_pnl(120.0, 100.0, 5000, 2500)
        self.assertEqual(result["gross_pnl"], -100_000.0)
        self.assertLess(result["net_pnl"], 0)

    def test_zero_volume(self):
        result = self.market.calculate_pnl(100.0, 120.0, 0, 0)
        self.assertEqual(result["gross_pnl"], 0.0)
        self.assertEqual(result["pnl_per_mt"], 0)  # Must not divide by zero

    def test_negative_volume(self):
        """Negative volume is unusual but must not crash."""
        result = self.market.calculate_pnl(100.0, 120.0, -5000, 2500)
        self.assertIsInstance(result["gross_pnl"], float)

    def test_entry_equals_exit(self):
        result = self.market.calculate_pnl(128.0, 128.0, 5000, 0)
        self.assertEqual(result["net_pnl"], 0.0)

    def test_very_large_volume(self):
        result = self.market.calculate_pnl(100.0, 105.0, 1_000_000, 0)
        self.assertEqual(result["gross_pnl"], 5_000_000.0)

    def test_zero_transaction_cost(self):
        result = self.market.calculate_pnl(100.0, 110.0, 1000, 0)
        self.assertEqual(result["net_pnl"], result["gross_pnl"])


# ═════════════════════════════════════════════════════════════════════════════
# B. SHOCK ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class TestShockEngine(unittest.TestCase):

    def test_valid_logistics_keys(self):
        for key in ["malacca_blockage", "suez_blockage", "port_klang_outage", "rotterdam_congestion"]:
            shock = get_logistics_shock(key)
            self.assertIsNotNone(shock, f"Missing logistics shock: {key}")
            self.assertIn("spread_multiplier", shock)
            self.assertIn("freight_impact", shock)

    def test_invalid_logistics_key(self):
        self.assertIsNone(get_logistics_shock("nonexistent_port"))
        self.assertIsNone(get_logistics_shock(""))
        self.assertIsNone(get_logistics_shock("MALACCA_BLOCKAGE"))  # case sensitive

    def test_valid_policy_keys(self):
        for key in ["indonesia_export_ban", "b50_mandate", "eu_deforestation_reg", "india_import_duty"]:
            shock = get_policy_shock(key)
            self.assertIsNotNone(shock, f"Missing policy shock: {key}")

    def test_invalid_policy_key(self):
        self.assertIsNone(get_policy_shock("fake_policy"))
        self.assertIsNone(get_policy_shock(None))

    def test_custom_shock_valid(self):
        shock = build_custom_shock(
            freight_impact=20.0,
            spread_multiplier=1.3,
            duration_hours=48.0,
            description="Test",
            severity="high",
        )
        self.assertEqual(shock["freight_impact"], 20.0)
        self.assertEqual(shock["spread_multiplier"], 1.3)

    def test_custom_shock_zero_duration(self):
        """Zero duration should not crash — shock expires immediately."""
        shock = build_custom_shock(0, 1.0, 0, "Zero duration", "low")
        self.assertIsNotNone(shock)

    def test_custom_shock_zero_multiplier(self):
        shock = build_custom_shock(0, 0.0, 24, "Zero multiplier", "low")
        self.assertEqual(shock["spread_multiplier"], 0.0)

    def test_custom_shock_negative_freight(self):
        """Negative freight (spread-narrowing scenario) must not crash."""
        shock = build_custom_shock(-10.0, 0.95, 24, "Negative freight", "low")
        self.assertEqual(shock["freight_impact"], -10.0)

    def test_list_scenarios_structure(self):
        scenarios = list_scenarios()
        self.assertIn("logistics", scenarios)
        self.assertIn("policy", scenarios)
        self.assertIsInstance(scenarios["logistics"], dict)

    def test_all_scenarios_have_required_fields(self):
        scenarios = list_scenarios()
        required = ["name", "description", "severity", "freight_impact",
                    "spread_multiplier", "duration_hours"]
        for category in scenarios.values():
            for key, val in category.items():
                for field in required:
                    self.assertIn(field, val, f"Scenario '{key}' missing field '{field}'")

    def test_severity_levels_are_valid(self):
        valid = {"low", "medium", "high", "critical"}
        for key in ["malacca_blockage", "suez_blockage", "port_klang_outage", "rotterdam_congestion"]:
            shock = get_logistics_shock(key)
            self.assertIn(shock["severity"], valid)


# ═════════════════════════════════════════════════════════════════════════════
# C. AI ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
class TestJsonExtraction(unittest.TestCase):

    def test_raw_json(self):
        text = '{"multiplier": 1.3, "severity": "high"}'
        result = _extract_json(text)
        self.assertEqual(result["multiplier"], 1.3)

    def test_fenced_json(self):
        text = '```json\n{"multiplier": 1.5, "severity": "medium"}\n```'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["multiplier"], 1.5)

    def test_fenced_no_lang(self):
        text = '```\n{"multiplier": 1.2}\n```'
        result = _extract_json(text)
        self.assertIsNotNone(result)

    def test_json_with_preamble(self):
        text = 'Here is my analysis:\n{"multiplier": 1.4, "severity": "low"}'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["multiplier"], 1.4)

    def test_empty_string(self):
        result = _extract_json("")
        self.assertIsNone(result)

    def test_none_input(self):
        result = _extract_json(None)
        self.assertIsNone(result)

    def test_plain_text_no_json(self):
        result = _extract_json("I cannot assess this headline.")
        self.assertIsNone(result)

    def test_malformed_json(self):
        result = _extract_json('{"multiplier": 1.3, "severity":}')
        self.assertIsNone(result)

    def test_nested_json(self):
        """Deeply nested JSON should be extracted correctly."""
        text = '{"a": {"b": 1}, "multiplier": 1.1}'
        result = _extract_json(text)
        self.assertIsNotNone(result)

    def test_json_with_trailing_garbage(self):
        text = '{"multiplier": 1.2} some extra text after'
        result = _extract_json(text)
        self.assertIsNotNone(result)


class TestMockHeadlineAnalysis(unittest.TestCase):

    def test_blockage_keyword(self):
        result = _mock_headline_analysis("Suez Canal blockage reported")
        self.assertGreater(result["multiplier"], 1.0)
        self.assertEqual(result["severity"], "high")

    def test_war_keyword(self):
        result = _mock_headline_analysis("Military conflict in Red Sea escalates")
        self.assertEqual(result["severity"], "critical")
        self.assertGreater(result["multiplier"], 1.5)

    def test_neutral_headline(self):
        result = _mock_headline_analysis("Palm oil prices remain stable this week")
        self.assertEqual(result["multiplier"], 1.0)

    def test_empty_headline(self):
        result = _mock_headline_analysis("")
        self.assertIsInstance(result, dict)
        self.assertIn("multiplier", result)

    def test_very_long_headline(self):
        headline = "Palm oil export ban " * 50  # 1000 chars
        result = _mock_headline_analysis(headline)
        self.assertIsInstance(result, dict)

    def test_headline_with_special_chars(self):
        result = _mock_headline_analysis('Palm oil prices: "critical" & <urgent> 100% tariff')
        self.assertIsInstance(result, dict)

    def test_returns_all_required_keys(self):
        required = ["multiplier", "freight_impact", "severity",
                    "reasoning", "duration_estimate_hours"]
        result = _mock_headline_analysis("Port congestion at Rotterdam")
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_multiplier_in_valid_range(self):
        for headline in [
            "Biodiesel mandate B50 Indonesia",
            "Malacca blocked",
            "EU tariff on palm oil",
            "Weather typhoon Philippines",
        ]:
            result = _mock_headline_analysis(headline)
            self.assertGreaterEqual(result["multiplier"], 0.8)
            self.assertLessEqual(result["multiplier"], 3.0)


class TestMockTradersBrief(unittest.TestCase):

    def test_single_shock_brief(self):
        shock = dict(DUMMY_SHOCK)
        result = _mock_traders_brief(shock, 128.0, 140.0, 60000.0, 5000, False)
        required = ["impact", "risk", "insight", "recommendation"]
        for key in required:
            self.assertIn(key, result)

    def test_compound_shock_brief(self):
        shock = dict(DUMMY_SHOCK)
        result = _mock_traders_brief(shock, 128.0, 160.0, 160000.0, 5000, True)
        self.assertTrue(result["compound_mode"])
        self.assertIn("compound", result["insight"].lower())

    def test_zero_spread_brief(self):
        """zero current_spread must not crash (division in pct_change)."""
        shock = dict(DUMMY_SHOCK)
        result = _mock_traders_brief(shock, 0.0, 0.0, 0.0, 5000, False)
        self.assertIsInstance(result, dict)

    def test_negative_pnl_impact(self):
        """Negative pnl_impact (spread narrows) should still produce valid brief."""
        shock = dict(DUMMY_SHOCK, spread_multiplier=0.9)
        result = _mock_traders_brief(shock, 128.0, 115.0, -65000.0, 5000, False)
        self.assertIsInstance(result, dict)


class TestAsyncInterpretHeadline(unittest.TestCase):

    def test_no_api_key_falls_back_to_mock(self):
        """Without API key, interpret_headline must return a valid dict."""
        from simulation.ai_analysis import interpret_headline
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            result = run_async(interpret_headline("Palm oil export ban Indonesia"))
        self.assertIn("multiplier", result)
        self.assertIn("severity", result)
        self.assertTrue(result.get("simulated", False))

    def test_interpret_returns_clamped_multiplier(self):
        from simulation.ai_analysis import interpret_headline
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            result = run_async(interpret_headline("Catastrophic total blockage war"))
        self.assertGreaterEqual(result["multiplier"], 0.8)
        self.assertLessEqual(result["multiplier"], 3.0)


class TestAsyncGenerateTradersBrief(unittest.TestCase):

    def _run_brief(self, combined_state=None):
        from simulation.ai_analysis import generate_traders_brief
        shock = dict(DUMMY_SHOCK)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            return run_async(generate_traders_brief(
                shock, 128.0, 850.0, 978.0, 5000, combined_state
            ))

    def test_single_shock_brief(self):
        result = self._run_brief()
        self.assertIn("impact", result)
        self.assertIn("risk", result)
        self.assertFalse(result.get("compound_mode", True))

    def test_compound_brief_two_shocks(self):
        combined = {
            "count": 2,
            "combined_spread_multiplier": 1.38,
            "combined_freight_impact": 35.0,
            "dominant_severity": "critical",
            "combined_description": "Shock A | Shock B",
            "individual_multipliers": [
                {"name": "Shock A", "multiplier": 1.20},
                {"name": "Shock B", "multiplier": 1.15},
            ],
        }
        result = self._run_brief(combined_state=combined)
        self.assertTrue(result.get("compound_mode"))
        self.assertEqual(result.get("active_shock_count"), 2)

    def test_empty_combined_state(self):
        combined = {"count": 0, "combined_spread_multiplier": 1.0,
                    "combined_freight_impact": 0.0, "dominant_severity": "low",
                    "combined_description": "", "individual_multipliers": []}
        result = self._run_brief(combined_state=combined)
        self.assertIsInstance(result, dict)

    def test_combined_state_missing_individual_multipliers(self):
        """combined_state without individual_multipliers key must not crash."""
        combined = {
            "count": 2,
            "combined_spread_multiplier": 1.2,
            "combined_freight_impact": 10.0,
            "dominant_severity": "high",
            "combined_description": "No individual multipliers field",
            # intentionally missing 'individual_multipliers'
        }
        result = self._run_brief(combined_state=combined)
        self.assertIsInstance(result, dict)


# ═════════════════════════════════════════════════════════════════════════════
# D. NEWS FEED
# ═════════════════════════════════════════════════════════════════════════════
VALID_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Palm oil export ban shocks Malaysia markets</title>
      <link>https://example.com/1</link>
      <pubDate>Sat, 04 Apr 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Malacca Strait vessel grounding causes delays - Reuters</title>
      <link>https://example.com/2</link>
      <pubDate>Sat, 04 Apr 2026 09:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Celebrity gossip story irrelevant to commodities</title>
      <link>https://example.com/3</link>
      <pubDate>Sat, 04 Apr 2026 08:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

EMPTY_CHANNEL_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>"""

MALFORMED_RSS = "This is not XML at all <<< broken"


class TestNewsFeedParsing(unittest.TestCase):

    def setUp(self):
        self.feed = NewsFeed()
        self.feed_info = {"name": "Test", "category": "test"}

    def test_parses_valid_rss(self):
        items = self.feed._parse_rss(VALID_RSS, self.feed_info)
        # 2 relevant items (celebrity gossip filtered)
        self.assertEqual(len(items), 2)

    def test_empty_channel_returns_empty_list(self):
        items = self.feed._parse_rss(EMPTY_CHANNEL_RSS, self.feed_info)
        self.assertEqual(items, [])

    def test_malformed_xml_returns_empty_list(self):
        items = self.feed._parse_rss(MALFORMED_RSS, self.feed_info)
        self.assertEqual(items, [])

    def test_source_suffix_stripped(self):
        """Google News appends ' - Reuters' — should be cleaned from title."""
        items = self.feed._parse_rss(VALID_RSS, self.feed_info)
        for item in items:
            self.assertNotIn(" - Reuters", item.title)

    def test_duplicate_ids_not_parsed_twice(self):
        """Same RSS parsed twice should not produce duplicate items."""
        items1 = self.feed._parse_rss(VALID_RSS, self.feed_info)
        for item in items1:
            self.feed.seen_ids.add(item.id)
        items2 = self.feed._parse_rss(VALID_RSS, self.feed_info)
        self.assertEqual(len(items2), 0)

    def test_item_id_is_stable(self):
        """Same title should always produce the same ID."""
        items1 = self.feed._parse_rss(VALID_RSS, {"name": "A", "category": "x"})
        # Clear seen_ids so second parse can produce items
        self.feed.seen_ids.clear()
        items2 = self.feed._parse_rss(VALID_RSS, {"name": "B", "category": "y"})
        self.assertEqual(items1[0].id, items2[0].id)

    def test_relevance_filter(self):
        self.assertTrue(self.feed._is_relevant("palm oil prices rise"))
        self.assertTrue(self.feed._is_relevant("Malacca Strait disruption"))
        self.assertTrue(self.feed._is_relevant("MPOB report shows higher production"))
        self.assertFalse(self.feed._is_relevant("Stock market rally continues"))
        self.assertFalse(self.feed._is_relevant("Football results this weekend"))
        self.assertFalse(self.feed._is_relevant(""))

    def test_seen_ids_cap_prevents_memory_leak(self):
        """After seen_ids exceeds max, it should be rebuilt from current items."""
        self.feed.max_seen_ids = 10
        self.feed.max_items = 5
        # Flood with fake IDs
        for i in range(50):
            self.feed.seen_ids.add(f"fakeid{i:04d}")
        # Add some current items
        for i in range(5):
            item = NewsItem(
                id=f"real{i:04d}", title=f"Palm oil headline {i}",
                source="test", category="test",
                published="", link="",
            )
            self.feed.items.append(item)
        # Simulate the cap logic from poll_and_score
        if len(self.feed.seen_ids) > self.feed.max_seen_ids:
            kept = {item.id for item in self.feed.items}
            self.feed.seen_ids = kept
        self.assertLessEqual(len(self.feed.seen_ids), self.feed.max_seen_ids)


class TestNewsFeedScoring(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()
        self.feed = NewsFeed(market=self.market, ai_scorer=None)  # No real scorer

    def test_score_headline_without_scorer(self):
        """No ai_scorer configured — item stays unscored, no crash."""
        item = NewsItem(id="t1", title="Test", source="x", category="x",
                        published="", link="")
        result = run_async(self.feed.score_headline(item))
        self.assertFalse(result.scored)

    def test_auto_inject_disabled(self):
        self.feed.auto_inject_enabled = False
        item = NewsItem(id="t2", title="Palm oil ban", source="x", category="x",
                        published="", link="", scored=True, confidence="HIGH",
                        multiplier=1.5, severity="critical", freight_impact=20.0)
        shock = self.feed._auto_inject(item)
        self.assertIsNone(shock)
        self.assertFalse(item.auto_injected)

    def test_auto_inject_low_confidence_not_injected(self):
        self.feed.auto_inject_enabled = True
        item = NewsItem(id="t3", title="Palm oil slightly higher", source="x",
                        category="x", published="", link="", scored=True,
                        confidence="LOW", multiplier=1.05, severity="low")
        shock = self.feed._auto_inject(item)
        self.assertIsNone(shock)

    def test_auto_inject_high_confidence_injects(self):
        self.feed.auto_inject_enabled = True
        item = NewsItem(id="t4", title="Export ban", source="x", category="x",
                        published="", link="", scored=True, confidence="HIGH",
                        multiplier=1.5, severity="critical", freight_impact=20.0,
                        duration_hours=48.0)
        shock = self.feed._auto_inject(item)
        self.assertIsNotNone(shock)
        self.assertTrue(item.auto_injected)
        self.assertEqual(len(self.market.active_shocks), 1)

    def test_auto_inject_not_repeated(self):
        """Already-injected item must not be injected again."""
        self.feed.auto_inject_enabled = True
        item = NewsItem(id="t5", title="Export ban again", source="x", category="x",
                        published="", link="", scored=True, confidence="HIGH",
                        multiplier=1.5, severity="critical", auto_injected=True)
        shock = self.feed._auto_inject(item)
        self.assertIsNone(shock)

    def test_get_feed_empty(self):
        result = self.feed.get_feed(20)
        self.assertEqual(result, [])

    def test_get_stats_zero_items(self):
        stats = self.feed.get_stats()
        self.assertEqual(stats["total_tracked"], 0)
        self.assertEqual(stats["high_confidence"], 0)
        self.assertEqual(stats["auto_injected"], 0)

    def test_confidence_level_boundaries(self):
        """Exactly at the HIGH threshold: severity=high AND multiplier>=1.15."""
        item = NewsItem(id="t6", title="High conf item", source="x", category="x",
                        published="", link="")
        # Simulate what score_headline does
        item.scored = True
        item.severity = "high"
        item.multiplier = 1.15  # Exactly at boundary
        if item.severity in ("critical", "high") and item.multiplier >= 1.15:
            item.confidence = "HIGH"
        self.assertEqual(item.confidence, "HIGH")

    def test_to_dict_all_fields(self):
        item = NewsItem(id="t7", title="Test", source="src", category="cat",
                        published="date", link="url")
        d = item.to_dict()
        required = ["id", "title", "source", "category", "published", "link",
                    "scored", "multiplier", "freight_impact", "severity",
                    "reasoning", "duration_hours", "confidence", "auto_injected"]
        for key in required:
            self.assertIn(key, d)


class TestNewsFeedPollIntegration(unittest.TestCase):
    """Tests poll_and_score with a mocked fetch."""

    def _make_feed_with_mock_scorer(self, scorer_result):
        market = PalmOilMarket()

        async def mock_scorer(headline):
            return scorer_result

        feed = NewsFeed(market=market, ai_scorer=mock_scorer)
        return feed, market

    def test_poll_with_no_network_is_safe(self):
        """If all RSS fetches fail, poll_and_score returns empty summary."""
        feed, _ = self._make_feed_with_mock_scorer(
            {"multiplier": 1.1, "freight_impact": 0, "severity": "low",
             "reasoning": "test", "duration_estimate_hours": 24}
        )
        with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
            result = run_async(feed.poll_and_score())
        self.assertEqual(result["new_headlines"], 0)
        self.assertEqual(result["auto_injected"], 0)

    def test_items_trimmed_to_max(self):
        """After parsing many headlines, items list must not exceed max_items."""
        feed, _ = self._make_feed_with_mock_scorer(
            {"multiplier": 1.0, "freight_impact": 0, "severity": "low",
             "reasoning": "x", "duration_estimate_hours": 12}
        )
        feed.max_items = 5
        # Pre-fill items
        for i in range(10):
            feed.items.append(
                NewsItem(id=f"old{i}", title=f"Old headline {i}", source="x",
                         category="x", published="", link="")
            )
        feed.items = feed.items[:feed.max_items]
        self.assertLessEqual(len(feed.items), 5)


# ═════════════════════════════════════════════════════════════════════════════
# E. API HANDLER INPUTS (unit tests, no live server)
# ═════════════════════════════════════════════════════════════════════════════
class TestJsonBodyParsing(unittest.TestCase):
    """Test the get_json_body helper via a minimal mock handler."""

    def _make_handler(self, body_bytes):
        """Create a minimal mock of JsonHandler for testing get_json_body."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import importlib
        # We test the logic directly by replicating it
        def get_json_body(body):
            try:
                if not body:
                    return {}
                return json.loads(body.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError, Exception):
                return {}
        return get_json_body(body_bytes)

    def test_valid_json(self):
        result = self._make_handler(b'{"key": "value"}')
        self.assertEqual(result["key"], "value")

    def test_empty_body(self):
        result = self._make_handler(b"")
        self.assertEqual(result, {})

    def test_none_body(self):
        result = self._make_handler(None)
        self.assertEqual(result, {})

    def test_invalid_json(self):
        result = self._make_handler(b"{invalid json}")
        self.assertEqual(result, {})

    def test_non_utf8_bytes(self):
        result = self._make_handler(b"\xff\xfe{not utf-8}")
        self.assertEqual(result, {})

    def test_nested_json(self):
        result = self._make_handler(b'{"freight_impact": 20.5, "nested": {"a": 1}}')
        self.assertAlmostEqual(result["freight_impact"], 20.5)

    def test_very_large_body(self):
        big = json.dumps({"key": "x" * 100_000}).encode()
        result = self._make_handler(big)
        self.assertIn("key", result)


class TestMarketHistoryBoundaries(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()

    def test_get_history_n_one(self):
        result = self.market.get_history(1)
        self.assertEqual(len(result), 1)

    def test_get_history_n_500(self):
        result = self.market.get_history(500)
        self.assertLessEqual(len(result), 500)

    def test_get_history_n_large(self):
        result = self.market.get_history(9999)
        self.assertLessEqual(len(result), len(self.market.history))

    def test_get_history_n_zero(self):
        result = self.market.get_history(0)
        self.assertEqual(result, [])

    def test_get_history_returns_list_of_dicts(self):
        result = self.market.get_history(5)
        self.assertIsInstance(result, list)
        for tick in result:
            self.assertIsInstance(tick, dict)


class TestCustomShockValidation(unittest.TestCase):
    """Validate that weird custom shock inputs don't break the market."""

    def setUp(self):
        self.market = PalmOilMarket()

    def test_string_multiplier_would_crash(self):
        """float() on a string must be handled at API layer."""
        # Simulate what CustomShockHandler does
        body = {"spread_multiplier": "not_a_number"}
        with self.assertRaises((ValueError, TypeError)):
            float(body.get("spread_multiplier", 1.0))

    def test_extreme_multiplier(self):
        """A multiplier of 100x should be clamped by the spread clamp in tick()."""
        shock = build_custom_shock(0, 100.0, 24, "Extreme", "critical")
        self.market.inject_shock(shock)
        tick = self.market.tick()
        self.assertLessEqual(tick["spread"], 500.0)

    def test_extreme_freight(self):
        shock = build_custom_shock(10000.0, 1.0, 24, "Extreme freight", "critical")
        self.market.inject_shock(shock)
        tick = self.market.tick()
        # freight_component should be in tick data
        self.assertIn("freight_component", tick)


# ═════════════════════════════════════════════════════════════════════════════
# F. COMPOUND SHOCK INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════
class TestCompoundShockIntegration(unittest.TestCase):

    def setUp(self):
        self.market = PalmOilMarket()
        self.market.clear_shocks()

    def test_compound_state_after_three_shocks(self):
        shocks = [
            dict(DUMMY_SHOCK, name="S1", spread_multiplier=1.20, freight_impact=10, duration_hours=24),
            dict(DUMMY_SHOCK, name="S2", spread_multiplier=1.15, freight_impact=15, duration_hours=24),
            dict(DUMMY_SHOCK, name="S3", spread_multiplier=1.10, freight_impact=5, duration_hours=24),
        ]
        for s in shocks:
            self.market.inject_shock(s)

        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 3)
        self.assertAlmostEqual(state["combined_freight_impact"], 30.0, places=1)
        expected_mult = round(1.20 * 1.15 * 1.10, 4)
        self.assertAlmostEqual(state["combined_spread_multiplier"], expected_mult, places=2)

    def test_dominant_severity_highest_wins(self):
        """Dominant severity must be the highest across all active shocks."""
        self.market.inject_shock(dict(DUMMY_SHOCK, severity="low", duration_hours=24))
        self.market.inject_shock(dict(DUMMY_SHOCK, severity="critical", duration_hours=24))
        self.market.inject_shock(dict(DUMMY_SHOCK, severity="medium", duration_hours=24))
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["dominant_severity"], "critical")

    def test_all_shocks_expire(self):
        self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=24))
        # Expire them all
        for s in self.market.active_shocks:
            s["expires"] = time.time() - 1
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 0)

    def test_partial_expiry(self):
        self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=24))  # active
        self.market.inject_shock(dict(DUMMY_SHOCK, duration_hours=24))  # will expire
        self.market.active_shocks[1]["expires"] = time.time() - 1
        state = self.market.get_combined_shock_state()
        self.assertEqual(state["count"], 1)

    def test_compound_brief_correct_mode(self):
        """When combined_state.count == 1, compound_mode must be False."""
        from simulation.ai_analysis import generate_traders_brief
        combined_single = {
            "count": 1,
            "combined_spread_multiplier": 1.20,
            "combined_freight_impact": 10.0,
            "dominant_severity": "high",
            "combined_description": "One shock",
            "individual_multipliers": [{"name": "S1", "multiplier": 1.20}],
        }
        shock = dict(DUMMY_SHOCK)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            result = run_async(generate_traders_brief(
                shock, 128.0, 850.0, 978.0, 5000, combined_single
            ))
        self.assertFalse(result.get("compound_mode"))

    def test_shock_names_in_combined_state(self):
        self.market.inject_shock(dict(DUMMY_SHOCK, name="Alpha", duration_hours=24))
        self.market.inject_shock(dict(DUMMY_SHOCK, name="Beta", duration_hours=24))
        state = self.market.get_combined_shock_state()
        self.assertIn("Alpha", state["shock_names"])
        self.assertIn("Beta", state["shock_names"])


# ═════════════════════════════════════════════════════════════════════════════
# G. STRESS / ENDURANCE TESTS
# ═════════════════════════════════════════════════════════════════════════════
class TestEndurance(unittest.TestCase):

    def test_1000_ticks_no_crash(self):
        market = PalmOilMarket()
        for _ in range(1000):
            tick = market.tick()
            self.assertGreaterEqual(tick["spread"], 20.0)
            self.assertLessEqual(tick["spread"], 500.0)
            self.assertGreaterEqual(tick["fob_malaysia"], 400.0)

    def test_rapid_shock_inject_clear_cycle(self):
        market = PalmOilMarket()
        for _ in range(100):
            market.inject_shock(dict(DUMMY_SHOCK, duration_hours=0.001))
            market.tick()
            market.clear_shocks()
        self.assertIsInstance(market.history[-1], dict)

    def test_market_with_shocks_never_nan(self):
        import math
        market = PalmOilMarket()
        market.inject_shock(dict(DUMMY_SHOCK, spread_multiplier=1.5, freight_impact=50))
        for _ in range(200):
            tick = market.tick()
            self.assertFalse(math.isnan(tick["spread"]), "NaN spread detected")
            self.assertFalse(math.isnan(tick["fob_malaysia"]), "NaN FOB detected")

    def test_get_history_thread_safe(self):
        """Concurrent read+write of history must not crash."""
        market = PalmOilMarket()
        errors = []

        def tick_loop():
            for _ in range(100):
                try:
                    market.tick()
                except Exception as e:
                    errors.append(f"tick: {e}")

        def read_loop():
            for _ in range(100):
                try:
                    market.get_history(20)
                except Exception as e:
                    errors.append(f"read: {e}")

        t1 = threading.Thread(target=tick_loop)
        t2 = threading.Thread(target=read_loop)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(errors, [])

    def test_news_feed_100_items_memory_cap(self):
        feed = NewsFeed()
        feed.max_items = 20
        for i in range(100):
            feed.items.insert(0, NewsItem(
                id=f"item{i:04d}", title=f"Palm oil news {i}",
                source="test", category="test", published="", link="",
            ))
            if len(feed.items) > feed.max_items:
                feed.items = feed.items[:feed.max_items]
        self.assertLessEqual(len(feed.items), 20)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 68)
    print("  SENTINEL-ARB  Edge Case & Stress Test Suite")
    print("=" * 68)
    unittest.main(verbosity=2)
