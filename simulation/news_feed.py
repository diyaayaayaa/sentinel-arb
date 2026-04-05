"""
Live News Feed → Haiku Risk Scorer → Auto-Shock Injector
==========================================================
Fetches palm-oil / commodity / geopolitics headlines from free RSS feeds,
scores them with Claude Haiku, and auto-injects HIGH+ confidence events
as shocks into the simulation engine.
"""

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from html import unescape

logger = logging.getLogger("sentinel-arb.news")

# ── RSS Feed Sources (all free, no API key needed) ──────────────────────
RSS_FEEDS = [
    # Reuters commodities
    {
        "name": "Reuters Commodities",
        "url": "https://news.google.com/rss/search?q=palm+oil+commodity+trade&hl=en-US&gl=US&ceid=US:en",
        "category": "commodities",
    },
    # Google News — palm oil specific
    {
        "name": "Palm Oil News",
        "url": "https://news.google.com/rss/search?q=%22palm+oil%22+OR+%22CPO%22+Malaysia+Indonesia&hl=en-US&gl=US&ceid=US:en",
        "category": "palm_oil",
    },
    # Shipping / logistics disruptions
    {
        "name": "Shipping & Trade",
        "url": "https://news.google.com/rss/search?q=shipping+disruption+OR+%22Malacca+Strait%22+OR+%22Suez+Canal%22+OR+%22port+congestion%22&hl=en-US&gl=US&ceid=US:en",
        "category": "logistics",
    },
    # Geopolitical / trade policy
    {
        "name": "Trade Policy",
        "url": "https://news.google.com/rss/search?q=export+ban+OR+biodiesel+mandate+OR+%22trade+war%22+OR+%22import+duty%22+commodity&hl=en-US&gl=US&ceid=US:en",
        "category": "policy",
    },
]

# Keywords that indicate a headline is relevant to palm oil basis trading
RELEVANCE_KEYWORDS = [
    "palm oil", "cpo", "crude palm", "vegetable oil", "edible oil",
    "malaysia", "indonesia", "rotterdam", "europe",
    "malacca", "suez", "shipping", "freight", "port",
    "export ban", "biodiesel", "b50", "b40", "deforestation",
    "import duty", "tariff", "trade war", "sanction",
    "typhoon", "flood", "drought", "el nino", "la nina",
    "refinery", "soybean oil", "rapeseed", "sunflower oil",
    "mpob", "gapki", "plantation",
]


@dataclass
class NewsItem:
    """A single scored news headline."""
    id: str
    title: str
    source: str
    category: str
    published: str
    link: str
    # AI scoring fields (populated after Haiku analysis)
    scored: bool = False
    multiplier: float = 1.0
    freight_impact: float = 0.0
    severity: str = "low"
    reasoning: str = ""
    duration_hours: float = 24.0
    confidence: str = "LOW"      # LOW / MEDIUM / HIGH
    auto_injected: bool = False
    score_timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "published": self.published,
            "link": self.link,
            "scored": self.scored,
            "multiplier": self.multiplier,
            "freight_impact": self.freight_impact,
            "severity": self.severity,
            "reasoning": self.reasoning,
            "duration_hours": self.duration_hours,
            "confidence": self.confidence,
            "auto_injected": self.auto_injected,
        }


class NewsFeed:
    """
    Manages RSS fetch → AI score → auto-inject pipeline.
    """

    def __init__(self, market=None, ai_scorer=None):
        self.market = market           # PalmOilMarket instance
        self.ai_scorer = ai_scorer     # interpret_headline function
        self.items: List[NewsItem] = []
        self.seen_ids: set = set()
        self.last_fetch: float = 0
        self.fetch_interval: int = 600  # seconds between RSS fetches (10 min)
        self.max_items: int = 50        # keep last 50 headlines
        self.max_seen_ids: int = 2000   # cap memory for seen headline IDs
        self.auto_inject_enabled: bool = True
        self._running: bool = False

    # ── RSS Fetching ──────────────────────────────────────────────────
    def _parse_rss(self, xml_text: str, feed_info: Dict) -> List[NewsItem]:
        """Parse RSS XML into NewsItem objects."""
        items = []
        try:
            root = ET.fromstring(xml_text)
            # Standard RSS 2.0
            for item_el in root.findall(".//item"):
                title_el = item_el.find("title")
                link_el = item_el.find("link")
                pub_el = item_el.find("pubDate")

                if title_el is None or not title_el.text:
                    continue

                title = unescape(title_el.text.strip())
                # Remove source suffix from Google News titles (e.g. " - Reuters")
                title_clean = re.sub(r'\s*-\s*[A-Za-z\s.]+$', '', title)

                link = link_el.text.strip() if link_el is not None and link_el.text else ""
                published = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

                # Generate stable ID from title hash
                item_id = hashlib.md5(title_clean.lower().encode()).hexdigest()[:12]

                if item_id in self.seen_ids:
                    continue

                # Relevance filter
                if not self._is_relevant(title_clean):
                    continue

                items.append(NewsItem(
                    id=item_id,
                    title=title_clean,
                    source=feed_info["name"],
                    category=feed_info["category"],
                    published=published,
                    link=link,
                ))
        except ET.ParseError as e:
            logger.warning("RSS parse error for %s: %s", feed_info["name"], e)
        return items

    def _is_relevant(self, title: str) -> bool:
        """Check if headline is relevant to palm oil / commodity trading."""
        title_lower = title.lower()
        return any(kw in title_lower for kw in RELEVANCE_KEYWORDS)

    async def fetch_feeds(self) -> List[NewsItem]:
        """Fetch all RSS feeds and return new (unscored) items."""
        import urllib.request
        import ssl

        new_items = []
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for feed in RSS_FEEDS:
            try:
                req = urllib.request.Request(
                    feed["url"],
                    headers={"User-Agent": "Sentinel-Arb/1.0 News Monitor"}
                )
                with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                    xml_text = resp.read().decode("utf-8", errors="replace")

                parsed = self._parse_rss(xml_text, feed)
                for item in parsed:
                    if item.id not in self.seen_ids:
                        self.seen_ids.add(item.id)
                        new_items.append(item)
                        logger.info("New headline [%s]: %s", feed["name"], item.title[:80])

            except Exception as e:
                logger.warning("RSS fetch failed for %s: %s", feed["name"], e)

        self.last_fetch = time.time()
        return new_items

    # ── AI Scoring ────────────────────────────────────────────────────
    async def score_headline(self, item: NewsItem) -> NewsItem:
        """Score a single headline with Claude Haiku."""
        if self.ai_scorer is None:
            return item

        try:
            analysis = await self.ai_scorer(item.title)
            item.scored = True
            item.multiplier = analysis.get("multiplier", 1.0)
            item.freight_impact = analysis.get("freight_impact", 0.0)
            item.severity = analysis.get("severity", "low")
            item.reasoning = analysis.get("reasoning", "")
            item.duration_hours = analysis.get("duration_estimate_hours", 24.0)
            item.score_timestamp = time.time()

            # Determine confidence level for auto-injection
            if item.severity in ("critical", "high") and item.multiplier >= 1.15:
                item.confidence = "HIGH"
            elif item.severity == "medium" and item.multiplier >= 1.1:
                item.confidence = "MEDIUM"
            else:
                item.confidence = "LOW"

            logger.info(
                "Scored [%s]: mult=%.2f sev=%s conf=%s — %s",
                item.id, item.multiplier, item.severity,
                item.confidence, item.title[:60],
            )
        except Exception as e:
            logger.error("Failed to score headline %s: %s", item.id, e)

        return item

    # ── Auto-Injection ────────────────────────────────────────────────
    def _auto_inject(self, item: NewsItem) -> Optional[Dict]:
        """
        If confidence is HIGH, auto-inject as a shock into the market.
        Returns the shock dict if injected, None otherwise.
        """
        if not self.auto_inject_enabled:
            return None
        if not self.market:
            return None
        if item.confidence != "HIGH":
            return None
        if item.auto_injected:
            return None

        from simulation.shock_engine import build_custom_shock

        shock = build_custom_shock(
            freight_impact=item.freight_impact,
            spread_multiplier=item.multiplier,
            duration_hours=item.duration_hours,
            description=f"[AUTO] {item.title}",
            severity=item.severity,
        )
        self.market.inject_shock(shock)
        item.auto_injected = True
        logger.info("AUTO-INJECTED shock from headline: %s (mult=%.2f)", item.title[:60], item.multiplier)
        return shock

    # ── Main Pipeline ─────────────────────────────────────────────────
    async def poll_and_score(self) -> Dict:
        """
        Single poll cycle: fetch → score → auto-inject.
        Returns a summary dict.
        """
        new_items = await self.fetch_feeds()
        scored_count = 0
        injected = []

        for item in new_items:
            await self.score_headline(item)
            scored_count += 1

            shock = self._auto_inject(item)
            if shock:
                injected.append({
                    "headline": item.title,
                    "shock": shock,
                    "item_id": item.id,
                })

            # Add to our items list
            self.items.insert(0, item)

            # Yield control back to the event loop between each headline
            # so Tornado can serve HTTP requests while scoring continues
            await asyncio.sleep(0.1)

        # Trim to max
        if len(self.items) > self.max_items:
            self.items = self.items[:self.max_items]

        # Cap seen_ids set to prevent memory growth over long uptime
        if len(self.seen_ids) > self.max_seen_ids:
            # Keep the most recent IDs by rebuilding from current items
            kept = {item.id for item in self.items}
            self.seen_ids = kept

        return {
            "new_headlines": len(new_items),
            "scored": scored_count,
            "auto_injected": len(injected),
            "injections": injected,
            "total_tracked": len(self.items),
        }

    # ── Background Loop ───────────────────────────────────────────────
    async def run_loop(self):
        """Background coroutine: poll RSS every `fetch_interval` seconds."""
        self._running = True
        logger.info("News feed loop started (interval=%ds)", self.fetch_interval)

        while self._running:
            try:
                result = await self.poll_and_score()
                if result["new_headlines"] > 0:
                    logger.info(
                        "News poll: %d new, %d scored, %d auto-injected",
                        result["new_headlines"], result["scored"], result["auto_injected"],
                    )
            except Exception as e:
                logger.error("News feed poll error: %s", e)

            await asyncio.sleep(self.fetch_interval)

    def stop(self):
        self._running = False

    # ── API helpers ───────────────────────────────────────────────────
    def get_feed(self, limit: int = 20) -> List[Dict]:
        """Return the latest scored headlines for the frontend."""
        return [item.to_dict() for item in self.items[:limit]]

    def get_stats(self) -> Dict:
        """Return feed statistics."""
        high = sum(1 for i in self.items if i.confidence == "HIGH")
        med = sum(1 for i in self.items if i.confidence == "MEDIUM")
        injected = sum(1 for i in self.items if i.auto_injected)
        return {
            "total_tracked": len(self.items),
            "high_confidence": high,
            "medium_confidence": med,
            "auto_injected": injected,
            "auto_inject_enabled": self.auto_inject_enabled,
            "last_fetch": self.last_fetch,
            "fetch_interval_seconds": self.fetch_interval,
            "feed_sources": [f["name"] for f in RSS_FEEDS],
        }
