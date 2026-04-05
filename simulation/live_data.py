"""
Live Market Data Fetcher
=========================
Multi-source live price feed for Palm Oil — ALL FREE, no API keys required:

  1. Primary:   World Bank Commodity API — proper JSON API, no key needed
  2. Secondary: Yahoo Finance ZL=F soybean oil proxy (0.87x ratio)
  3. Tertiary:  FRED St. Louis Fed (monthly CPO series)
  4. Fallback:  Simulated data from OU process

Prices are cached and refreshed at configurable intervals.
The fetcher runs in a background thread to avoid blocking the event loop.
"""

import os
import time
import json
import re
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import urllib.request
import ssl

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# SSL context that works on macOS without cert issues
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _url_get(url: str, headers: dict = None, timeout: int = 12) -> str:
    """
    Fetch a URL. Returns text on success, raises on failure.
    Uses requests if available (better SSL handling), falls back to urllib.
    """
    hdrs = headers or {"User-Agent": "Mozilla/5.0 Sentinel-Arb/1.0"}
    if HAS_REQUESTS:
        resp = requests.get(url, headers=hdrs, timeout=timeout, verify=False)
        resp.raise_for_status()
        return resp.text
    else:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return resp.read().decode("utf-8", errors="replace")

logger = logging.getLogger("sentinel-arb.live_data")


class LivePriceFetcher:
    """
    Fetches real Palm Oil prices from multiple FREE sources with caching.

    Price model:
      - FOB Malaysia (CPO): Direct market price
      - CIF Rotterdam: FOB + freight + insurance + refining + basis premium
      - When only one price is available, derives the other using the spread model
    """

    # Freight & logistics model (used to derive CIF from FOB or vice versa)
    FREIGHT_ESTIMATE    = 55.0   # $/MT Malaysia -> Rotterdam
    INSURANCE_ESTIMATE  = 8.0    # $/MT
    REFINING_MARGIN     = 45.0   # $/MT (crude -> refined)
    BASIS_PREMIUM       = 20.0   # $/MT quality/geopolitical
    TOTAL_SPREAD_ESTIMATE = (
        FREIGHT_ESTIMATE + INSURANCE_ESTIMATE + REFINING_MARGIN + BASIS_PREMIUM
    )

    # Soybean oil to palm oil correlation factor
    SOYBEAN_TO_PALM_RATIO = 0.87

    def __init__(self, refresh_interval_seconds: int = 300):
        self.refresh_interval = refresh_interval_seconds
        self.cache: Dict = {}
        self.last_fetch_time: float = 0
        self.last_fetch_source: str = "none"
        self.is_live: bool = False
        self.fetch_error: Optional[str] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()
        logger.info("Live price fetcher started (refresh every %ds)", self.refresh_interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_prices(self) -> Dict:
        with self._lock:
            if self.cache:
                return self.cache.copy()
        return {
            "fob_malaysia": None,
            "cif_rotterdam": None,
            "spread": None,
            "source": "initializing",
            "is_live": False,
            "last_updated": datetime.utcnow().isoformat(),
            "market_status": "unknown",
            "error": "Fetching initial prices...",
        }

    # ── Background loop ──────────────────────────────────────────────────
    def _fetch_loop(self):
        self._do_fetch()
        while self._running:
            time.sleep(self.refresh_interval)
            if self._running:
                self._do_fetch()

    def _do_fetch(self):
        result = None

        # Try each source in priority order
        for fetch_fn in [
            self._fetch_world_bank,
            self._fetch_yahoo_proxy,
            self._fetch_fred,
            self._fetch_indexmundi,
        ]:
            try:
                result = fetch_fn()
                if result:
                    break
            except Exception as e:
                logger.debug("%s failed: %s", fetch_fn.__name__, e)

        if result:
            with self._lock:
                self.cache = result
                self.last_fetch_time = time.time()
                self.is_live = True
                self.last_fetch_source = result.get("source", "unknown")
                self.fetch_error = None
            logger.info(
                "Price update from %s: FOB=$%.2f, CIF=$%.2f",
                result["source"],
                result.get("fob_malaysia", 0),
                result.get("cif_rotterdam", 0),
            )
        else:
            with self._lock:
                self.fetch_error = "All live sources unavailable"
                self.is_live = False
                if not self.cache or not self.cache.get("is_live"):
                    self.cache = {
                        "fob_malaysia": None,
                        "cif_rotterdam": None,
                        "spread": None,
                        "source": "unavailable",
                        "is_live": False,
                        "last_updated": datetime.utcnow().isoformat(),
                        "market_status": "unknown",
                        "error": "All live sources unavailable — using simulation",
                    }
            logger.warning("All live price sources failed — using simulation fallback")

    # ==================================================================
    # SOURCE 1: World Bank Commodity Price API — FREE, no key needed
    # ==================================================================
    # API: https://api.worldbank.org/v2/en/indicator/PPOILUSDM
    # Returns monthly palm oil price in USD/MT — most reliable free JSON API
    # ==================================================================
    def _fetch_world_bank(self) -> Optional[Dict]:
        """World Bank monthly CPO price — proper REST JSON, always free."""
        try:
            # Per-page=1 gets the latest observation
            url = (
                "https://api.worldbank.org/v2/en/indicator/PPOILUSDM"
                "?format=json&per_page=1&mrv=1&gapfill=Y"
            )
            text = _url_get(url, timeout=10)
            data = json.loads(text)
            # World Bank returns [metadata, [observations]]
            if isinstance(data, list) and len(data) >= 2:
                obs_list = data[1]
                if obs_list:
                    value = obs_list[0].get("value")
                    if value and 300 < float(value) < 3000:
                        fob_price = float(value)
                        cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                        return self._build_result(
                            fob=fob_price, cif=cif_price,
                            source="World Bank Commodity API",
                        )
        except Exception as e:
            logger.debug("World Bank fetch failed: %s", e)
        return None

    # ==================================================================
    # SOURCE 2: FRED (Federal Reserve Economic Data) — 100% FREE
    # ==================================================================
    # Series PPOILUSDM = Global price of Palm Oil, monthly, USD/MT
    # No API key needed for the JSON observation endpoint
    # ==================================================================
    def _fetch_fred(self) -> Optional[Dict]:
        """Fetch palm oil price from FRED (St. Louis Fed) — completely free."""
        try:
            fred_api_key = os.getenv("FRED_API_KEY", "")

            if fred_api_key:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id=PPOILUSDM&api_key={fred_api_key}"
                    f"&file_type=json&sort_order=desc&limit=1"
                )
                text = _url_get(url)
                if text:
                    data = json.loads(text)
                    obs = data.get("observations", [])
                    if obs and obs[0].get("value", ".") != ".":
                        fob_price = float(obs[0]["value"])
                        if 300 < fob_price < 3000:
                            cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                            return self._build_result(
                                fob=fob_price, cif=cif_price,
                                source="FRED (Federal Reserve)",
                            )

            # Fallback: scrape the FRED series page directly (no key needed)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            text = _url_get(
                "https://fred.stlouisfed.org/series/PPOILUSDM",
                headers=headers,
            )
            if text:
                match = re.search(
                    r'(\d{3,4}\.\d{2})\s*(?:Dollars per Metric Ton|USD/MT)', text
                )
                if match:
                    fob_price = float(match.group(1))
                    if 300 < fob_price < 3000:
                        cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                        return self._build_result(
                            fob=fob_price, cif=cif_price,
                            source="FRED (Federal Reserve)",
                        )
                # Look in embedded script JSON
                price_matches = re.findall(
                    r'"value"\s*:\s*"(\d{3,4}\.\d+)"', text
                )
                if price_matches:
                    fob_price = float(price_matches[-1])
                    if 300 < fob_price < 3000:
                        cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                        return self._build_result(
                            fob=fob_price, cif=cif_price,
                            source="FRED (Federal Reserve)",
                        )

        except Exception as e:
            logger.debug("FRED fetch failed: %s", e)

        return None

    # ==================================================================
    # SOURCE 2: Yahoo Finance — soybean oil futures as proxy — FREE
    # ==================================================================
    def _fetch_yahoo_proxy(self) -> Optional[Dict]:
        """
        Soybean oil futures (ZL=F) from Yahoo Finance.
        Palm oil typically trades at ~87% of soybean oil price.
        """
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/ZL%3DF?interval=1d&range=5d"
            text = _url_get(url, headers={"User-Agent": "Mozilla/5.0"})

            if text:
                data = json.loads(text)
                result = data.get("chart", {}).get("result", [])
                if result:
                    meta = result[0].get("meta", {})
                    regular_price = meta.get("regularMarketPrice", 0)

                    if regular_price > 0:
                        soybean_usd_per_mt = regular_price * 22.0462 / 100
                        fob_price = soybean_usd_per_mt * self.SOYBEAN_TO_PALM_RATIO

                        if 300 < fob_price < 3000:
                            cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                            return self._build_result(
                                fob=fob_price, cif=cif_price,
                                source="Yahoo Finance (soybean oil proxy)",
                            )

        except Exception as e:
            logger.debug("Yahoo Finance proxy failed: %s", e)

        return None

    # ==================================================================
    # SOURCE 3: IndexMundi — historical commodity prices — FREE
    # ==================================================================
    def _fetch_indexmundi(self) -> Optional[Dict]:
        """Scrape latest palm oil price from IndexMundi (free, no key)."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            text = _url_get(
                "https://www.indexmundi.com/commodities/?commodity=palm-oil",
                headers=headers,
            )

            if text:
                if HAS_BS4:
                    soup = BeautifulSoup(text, "html.parser")
                    tables = soup.find_all("table", class_="tblData")
                    for table in tables:
                        rows = table.find_all("tr")
                        if rows:
                            last_row = rows[-1]
                            cells = last_row.find_all("td")
                            if len(cells) >= 2:
                                try:
                                    price_text = cells[1].get_text(strip=True).replace(",", "")
                                    fob_price = float(price_text)
                                    if 300 < fob_price < 3000:
                                        cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                                        return self._build_result(
                                            fob=fob_price, cif=cif_price,
                                            source="IndexMundi",
                                        )
                                except (ValueError, IndexError):
                                    continue

                # Regex fallback
                match = re.search(
                    r'Palm\s+Oil.*?(\d{3,4}(?:\.\d{1,2})?)\s*(?:USD|dollar)',
                    text, re.IGNORECASE | re.DOTALL
                )
                if match:
                    fob_price = float(match.group(1))
                    if 300 < fob_price < 3000:
                        cif_price = fob_price + self.TOTAL_SPREAD_ESTIMATE
                        return self._build_result(
                            fob=fob_price, cif=cif_price,
                            source="IndexMundi",
                        )

        except Exception as e:
            logger.debug("IndexMundi fetch failed: %s", e)

        return None

    # ── Helpers ───────────────────────────────────────────────────────────
    def _build_result(self, fob: float, cif: float, source: str) -> Dict:
        spread = cif - fob
        now = datetime.utcnow()

        # Bursa Malaysia hours: 10:30-18:00 MYT = 02:30-10:00 UTC
        utc_hour = now.hour
        is_weekday = now.weekday() < 5
        market_open = is_weekday and (2 <= utc_hour <= 12)

        return {
            "fob_malaysia": round(fob, 2),
            "cif_rotterdam": round(cif, 2),
            "spread": round(spread, 2),
            "source": source,
            "is_live": True,
            "last_updated": now.isoformat(),
            "market_status": "open" if market_open else "closed",
            "error": None,
        }


# ── Singleton ────────────────────────────────────────────────────────────
_fetcher: Optional[LivePriceFetcher] = None


def get_fetcher() -> LivePriceFetcher:
    global _fetcher
    if _fetcher is None:
        refresh = int(os.getenv("PRICE_REFRESH_SECONDS", "300"))
        _fetcher = LivePriceFetcher(refresh_interval_seconds=refresh)
    return _fetcher
