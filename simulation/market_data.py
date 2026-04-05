"""
Market Data Engine — Hybrid Live + Simulation
===============================================
When live data is available (from LivePriceFetcher), the OU stochastic
process is re-anchored to the real market price so the chart tracks
reality.  Micro-volatility (intra-refresh jitter) is layered on top to
give a smooth, ticking feel between 5-minute refreshes.

When live data is unavailable (API down, market closed, no network),
the engine falls back to pure simulation seamlessly.

Spread model:
  Spread = CIF Rotterdam − FOB Malaysia
         = freight + insurance + refining margin + basis premium + shock
"""

import numpy as np
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger("sentinel-arb.market")

# Maximum ticks kept in memory (covers ~30 min at 3-sec interval + 90 days history)
MAX_HISTORY = 600


# ── Ornstein-Uhlenbeck process ───────────────────────────────────────────
class OrnsteinUhlenbeck:
    """Mean-reverting stochastic process for commodity prices."""

    def __init__(self, mu: float, theta: float, sigma: float, dt: float = 1/252):
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt

    def step(self, x: float) -> float:
        dx = self.theta * (self.mu - x) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * np.random.normal()
        return x + dx + diffusion


# ── Main market class ────────────────────────────────────────────────────
class PalmOilMarket:
    """
    Dual-mode Palm Oil market engine.
    Thread-safe: tick() and inject_shock() can be called from different threads.
    """

    # Baseline parameters
    FOB_MEAN        = 850.0
    FOB_THETA       = 0.15
    FOB_SIGMA       = 120.0

    FREIGHT_BASE    = 55.0
    INSURANCE_BASE  = 8.0
    REFINING_MARGIN = 45.0
    BASIS_PREMIUM   = 20.0

    SPREAD_SIGMA    = 15.0

    # Micro-jitter (keeps chart ticking between live refreshes)
    JITTER_FOB      = 1.5
    JITTER_SPREAD   = 0.6

    def __init__(self):
        self._lock = threading.Lock()

        self.fob_process = OrnsteinUhlenbeck(
            mu=self.FOB_MEAN, theta=self.FOB_THETA, sigma=self.FOB_SIGMA,
        )
        self.spread_process = OrnsteinUhlenbeck(
            mu=self._base_spread(), theta=0.20, sigma=self.SPREAD_SIGMA,
        )

        # State
        self.fob_price   = float(self.FOB_MEAN + np.random.normal(0, 30))
        self.spread      = float(self._base_spread() + np.random.normal(0, 5))
        self.history: List[Dict]      = []
        self.active_shocks: List[Dict] = []

        # Live data anchor
        self._live_fob:     Optional[float] = None
        self._live_spread:  Optional[float] = None
        self._live_source:  str             = "simulated"
        self._live_updated: Optional[str]   = None
        self._is_live:      bool            = False

        # Pre-populate 90 trading-day history
        self._generate_history(days=90)

    # ── Spread helpers ───────────────────────────────────────────────────
    def _base_spread(self) -> float:
        return (
            self.FREIGHT_BASE
            + self.INSURANCE_BASE
            + self.REFINING_MARGIN
            + self.BASIS_PREMIUM
        )

    def _seasonal_factor(self, date: datetime) -> float:
        month = date.month
        if month in (10, 11, 12, 1, 2):
            return -0.03
        elif month in (6, 7, 8):
            return 0.04
        return 0.0

    # ── Shock logic ──────────────────────────────────────────────────────
    def _shock_adjustment(self) -> Dict[str, float]:
        freight_add = 0.0
        spread_mult = 1.0
        now = time.time()
        still_active = []
        for shock in self.active_shocks:
            if now < shock.get("expires", now + 1):
                freight_add += shock.get("freight_impact", 0.0)
                spread_mult *= shock.get("spread_multiplier", 1.0)
                still_active.append(shock)
        self.active_shocks = still_active
        return {"freight_add": freight_add, "spread_mult": spread_mult}

    def inject_shock(self, shock: Dict):
        with self._lock:
            shock["expires"] = time.time() + shock.get("duration_hours", 24) * 3600
            self.active_shocks.append(shock)
            logger.info("Shock injected: %s (%.2fx spread, %sh)", shock.get("name"), shock.get("spread_multiplier", 1.0), shock.get("duration_hours", 24))

    def get_combined_shock_state(self) -> Dict:
        """
        Returns the true compounded effect of ALL currently active shocks.
        This is what Haiku should reason about — not just the latest shock.
        Thread-safe: holds lock while reading active_shocks.
        """
        now = time.time()
        with self._lock:
            active = [s for s in self.active_shocks if now < s.get("expires", now + 1)]

        if not active:
            return {
                "count": 0,
                "shocks": [],
                "combined_freight_impact": 0.0,
                "combined_spread_multiplier": 1.0,
                "combined_description": "No active shocks.",
                "dominant_severity": "low",
            }

        combined_freight = sum(s.get("freight_impact", 0.0) for s in active)
        combined_mult = 1.0
        for s in active:
            combined_mult *= s.get("spread_multiplier", 1.0)

        # Dominant severity = highest of all active shocks
        sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        dominant_sev = max(
            (s.get("severity", "low") for s in active),
            key=lambda x: sev_rank.get(x, 0),
        )

        names = [s.get("name", "Unknown") for s in active]
        descriptions = [s.get("description", "") for s in active]

        return {
            "count": len(active),
            "shocks": active,
            "shock_names": names,
            "combined_freight_impact": round(combined_freight, 2),
            "combined_spread_multiplier": round(combined_mult, 4),
            "combined_description": " | ".join(descriptions),
            "dominant_severity": dominant_sev,
            "individual_multipliers": [
                {"name": s.get("name"), "multiplier": s.get("spread_multiplier", 1.0)}
                for s in active
            ],
        }

    def clear_shocks(self):
        with self._lock:
            self.active_shocks = []

    # ── Live data anchor ─────────────────────────────────────────────────
    def anchor_to_live(self, live_prices: Dict):
        """Re-centres OU processes on real market data."""
        fob    = live_prices.get("fob_malaysia")
        cif    = live_prices.get("cif_rotterdam")
        source = live_prices.get("source", "unknown")
        is_live = live_prices.get("is_live", False)

        if fob and cif and is_live and 300 < fob < 3000:
            spread = cif - fob
            spread = max(20.0, min(500.0, spread))  # clamp
            with self._lock:
                self._live_fob     = float(fob)
                self._live_spread  = float(spread)
                self._live_source  = source
                self._live_updated = live_prices.get("last_updated")
                self._is_live      = True
                self.fob_process.mu    = float(fob)
                self.spread_process.mu = float(spread)
                self.fob_price = fob + np.random.normal(0, self.JITTER_FOB)
                self.spread    = spread + np.random.normal(0, self.JITTER_SPREAD)
            logger.info("Anchored to live: FOB=$%.2f Spread=$%.2f [%s]", fob, spread, source)
        else:
            with self._lock:
                self._is_live      = False
                self._live_source  = "simulated (live unavailable)"

    # ── Tick ─────────────────────────────────────────────────────────────
    def tick(self, date: Optional[datetime] = None) -> Dict:
        if date is None:
            date = datetime.utcnow()

        with self._lock:
            try:
                seasonal  = self._seasonal_factor(date)
                shock_adj = self._shock_adjustment()

                if self._is_live and self._live_fob:
                    # LIVE MODE: micro-jitter around real anchor
                    self.fob_price += np.random.normal(0, self.JITTER_FOB)
                    self.fob_price += 0.3 * (self._live_fob - self.fob_price)

                    base_target    = (self._live_spread or self._base_spread()) + shock_adj["freight_add"]
                    shocked_target = base_target * shock_adj["spread_mult"]
                    self.spread   += np.random.normal(0, self.JITTER_SPREAD)
                    self.spread   += 0.3 * (shocked_target - self.spread)
                else:
                    # SIM MODE: full OU evolution
                    self.fob_price  = self.fob_process.step(self.fob_price)
                    self.fob_price *= (1 + seasonal * (1/252))

                    base_spread    = self._base_spread() + shock_adj["freight_add"]
                    shocked_target = base_spread * shock_adj["spread_mult"]
                    self.spread_process.mu = shocked_target
                    self.spread = self.spread_process.step(self.spread)

                # Hard clamps — prevents any runaway values
                self.fob_price = max(min(float(self.fob_price), 2000.0), 400.0)
                self.spread    = max(min(float(self.spread),    500.0),  20.0)

                cif_price = self.fob_price + self.spread

                tick_data = {
                    "timestamp":         date.isoformat(),
                    "fob_malaysia":      round(self.fob_price, 2),
                    "cif_rotterdam":     round(cif_price, 2),
                    "spread":            round(self.spread, 2),
                    "base_spread":       round(self._base_spread(), 2),
                    "freight_component": round(self.FREIGHT_BASE + shock_adj["freight_add"], 2),
                    "shock_active":      len(self.active_shocks) > 0,
                    "shock_count":       len(self.active_shocks),
                    "data_mode":         "live" if self._is_live else "simulated",
                    "data_source":       self._live_source,
                    "live_updated":      self._live_updated,
                }

                self.history.append(tick_data)
                # Trim: keep 90-day baseline + latest live ticks, never exceed MAX_HISTORY
                if len(self.history) > MAX_HISTORY:
                    self.history = self.history[-MAX_HISTORY:]

                return tick_data

            except Exception as e:
                logger.error("tick() error: %s", e)
                # Return last known state rather than crashing
                if self.history:
                    return self.history[-1]
                return {
                    "timestamp": date.isoformat(),
                    "fob_malaysia": self.FOB_MEAN,
                    "cif_rotterdam": self.FOB_MEAN + self._base_spread(),
                    "spread": self._base_spread(),
                    "base_spread": self._base_spread(),
                    "freight_component": self.FREIGHT_BASE,
                    "shock_active": False,
                    "shock_count": 0,
                    "data_mode": "simulated",
                    "data_source": "error_fallback",
                    "live_updated": None,
                }

    # ── History ──────────────────────────────────────────────────────────
    def _generate_history(self, days: int = 90):
        """Pre-populate historical baseline on startup."""
        start = datetime.utcnow() - timedelta(days=days)
        for i in range(days):
            date = start + timedelta(days=i)
            if date.weekday() < 5:  # weekdays only
                self.tick(date)

    def get_history(self, n: int = 90) -> List[Dict]:
        if n <= 0:
            return []
        with self._lock:
            return list(self.history[-n:])

    # ── P&L ──────────────────────────────────────────────────────────────
    def calculate_pnl(
        self,
        entry_spread: float,
        exit_spread: float,
        volume_mt: float = 5000,
        transaction_cost: float = 2500,
    ) -> Dict:
        gross_pnl = (exit_spread - entry_spread) * volume_mt
        net_pnl   = gross_pnl - transaction_cost
        return {
            "entry_spread":     round(entry_spread, 2),
            "exit_spread":      round(exit_spread, 2),
            "spread_change":    round(exit_spread - entry_spread, 2),
            "volume_mt":        volume_mt,
            "gross_pnl":        round(gross_pnl, 2),
            "transaction_cost": transaction_cost,
            "net_pnl":          round(net_pnl, 2),
            "pnl_per_mt":       round(net_pnl / volume_mt, 2) if volume_mt else 0,
        }
