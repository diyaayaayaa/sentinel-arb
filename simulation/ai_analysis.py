"""
Claude Haiku AI Analysis Module
================================
Two core functions:
  1. Interpret a news headline → Basis Volatility Multiplier
  2. Generate a Trader's Brief for any shock/event
"""

import os
import re
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger("sentinel-arb.ai")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Try multiple model names in case one isn't available
MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]


def _get_client():
    """Initialize Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("No ANTHROPIC_API_KEY configured — using simulated analysis")
        return None
    if not HAS_ANTHROPIC:
        logger.warning("anthropic package not installed — using simulated analysis")
        return None
    return anthropic.Anthropic(api_key=api_key)


def _extract_json(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from model output.
    Handles: raw JSON, ```json fences, ```fences, mixed text + JSON.
    """
    if not text:
        return None

    # 1. Try parsing the raw text directly
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences: ```json ... ``` or ``` ... ```
    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find the first { ... } block in the text
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _call_haiku_sync(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """
    Synchronous Haiku call — runs inside a thread executor so it
    never blocks the Tornado event loop.
    """
    client = _get_client()
    if not client:
        return None

    last_error = None
    for model in MODELS:
        try:
            logger.info("Calling %s...", model)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            logger.info("Haiku response (%s): %s", model, text[:200])
            return text
        except anthropic.NotFoundError:
            logger.debug("Model %s not found, trying next...", model)
            last_error = f"Model {model} not found"
            continue
        except anthropic.AuthenticationError as e:
            logger.error("API key invalid: %s", e)
            last_error = f"Authentication failed: {e}"
            break
        except Exception as e:
            logger.error("Haiku call failed (%s): %s", model, e)
            last_error = str(e)
            continue

    logger.warning("All Haiku model attempts failed. Last error: %s", last_error)


async def _call_haiku(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """
    Async wrapper: runs the blocking Anthropic SDK call in a thread
    so the Tornado event loop stays free to serve HTTP requests.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: _call_haiku_sync(prompt, max_tokens)
    )


async def interpret_headline(headline: str) -> Dict:
    """
    Interpret a news headline and output a Basis Volatility Multiplier.
    """
    prompt = f"""You are a senior commodities risk analyst specializing in palm oil
trade flows between Southeast Asia and Europe.

Analyze this news headline and assess its impact on the Palm Oil basis spread
(CIF Rotterdam minus FOB Malaysia):

HEADLINE: "{headline}"

Respond with ONLY a JSON object (no markdown, no explanation, no code fences):
{{
    "multiplier": <float between 0.8 and 3.0, where 1.0 = no impact, >1.0 = spread widens, <1.0 = spread narrows>,
    "freight_impact": <float, estimated additional freight cost in $/MT, 0 if not freight-related>,
    "severity": "<low|medium|high|critical>",
    "reasoning": "<1-2 sentences explaining why this event impacts the Malaysia-Rotterdam palm oil spread>",
    "duration_estimate_hours": <float, how long the price impact likely persists>
}}

Consider: shipping routes (Malacca Strait, Suez Canal), export policies (Indonesia, Malaysia),
biodiesel mandates, EU regulations, weather events, port infrastructure, and demand shifts."""

    text = await _call_haiku(prompt, max_tokens=500)

    if text:
        result = _extract_json(text)
        if result:
            # Validate and clamp values
            result["multiplier"] = max(0.8, min(3.0, float(result.get("multiplier", 1.0))))
            result["freight_impact"] = max(0, float(result.get("freight_impact", 0)))
            result["duration_estimate_hours"] = max(1, float(result.get("duration_estimate_hours", 24)))
            if "severity" not in result:
                result["severity"] = "medium"
            if "reasoning" not in result:
                result["reasoning"] = "AI analysis completed."
            return result
        else:
            logger.warning("Could not parse JSON from Haiku response: %s", text[:300])

    # Fallback to mock
    return _mock_headline_analysis(headline)


async def generate_traders_brief(
    shock_data: Dict,
    current_spread: float,
    fob_price: float,
    cif_price: float,
    position_mt: float = 5000,
    combined_state: Dict = None,
) -> Dict:
    """
    Generate a Trader's Brief after a shock event.

    If combined_state is provided (from market.get_combined_shock_state()),
    Haiku reasons over the TRUE compounded effect of ALL active shocks,
    not just the latest one. This is the correct behaviour when stacking shocks.
    """
    # ── Determine whether we are in single-shock or multi-shock mode ──
    is_compound = combined_state and combined_state.get("count", 0) > 1

    if is_compound:
        # Use the compounded multiplier and freight from all active shocks
        eff_mult    = combined_state["combined_spread_multiplier"]
        eff_freight = combined_state["combined_freight_impact"]
        severity    = combined_state["dominant_severity"]
    else:
        eff_mult    = shock_data.get("spread_multiplier", 1.0)
        eff_freight = shock_data.get("freight_impact", 0.0)
        severity    = shock_data.get("severity", "medium")

    spread_change = current_spread * (eff_mult - 1.0)
    new_spread    = current_spread + spread_change
    pnl_impact    = spread_change * position_mt

    # ── Build the prompt ──────────────────────────────────────────────
    if is_compound:
        shock_block = f"""COMPOUND SHOCK STATE ({combined_state['count']} active shocks):
{chr(10).join(f"  - {item['name']}: {item['multiplier']:.2f}x" for item in combined_state.get('individual_multipliers', []))}

COMBINED EFFECT (what the market is actually feeling right now):
- Combined Spread Multiplier: {eff_mult:.4f}x  (individual multipliers compounded)
- Combined Freight Impact: +${eff_freight:.2f}/MT  (individual freight impacts summed)
- Dominant Severity: {severity.upper()}
- Compounded Scenario: {combined_state.get('combined_description', 'Multiple concurrent disruptions')}

NEWLY INJECTED SHOCK (triggered this brief):
- Name: {shock_data.get('name', 'Unknown')}
- Description: {shock_data.get('description', 'N/A')}
- Its isolated multiplier: {shock_data.get('spread_multiplier', 1.0):.2f}x"""
    else:
        shock_block = f"""SHOCK EVENT:
- Name: {shock_data.get('name', 'Unknown Event')}
- Description: {shock_data.get('description', 'N/A')}
- Spread Multiplier: {eff_mult:.2f}x
- Freight Impact: +${eff_freight:.2f}/MT
- Severity: {severity.upper()}"""

    prompt = f"""You are a senior commodities trader generating an urgent risk brief.

MARKET STATE:
- FOB Malaysia: ${fob_price:.2f}/MT
- CIF Rotterdam: ${cif_price:.2f}/MT
- Current Spread: ${current_spread:.2f}/MT
- Position: {position_mt:,.0f} MT (long spread)

{shock_block}

QUANTITATIVE IMPACT:
- Estimated New Spread: ${new_spread:.2f}/MT ({spread_change:+.2f}, {((eff_mult-1)*100):+.1f}%)
- Estimated P&L Impact: ${pnl_impact:+,.0f} for {position_mt:,.0f} MT position

{"IMPORTANT: You must reason about the COMPOUND effect of ALL active shocks together, not just the most recent one. The market is simultaneously experiencing all the shocks listed above." if is_compound else ""}

Respond with ONLY a JSON object (no markdown, no explanation, no code fences):
{{
    "impact": "<Quantitative spread shift in $/MT and % — reference the {'compounded' if is_compound else 'single'} multiplier explicitly>",
    "risk": "<P&L sensitivity for {position_mt:,.0f} MT — best/base/worst case in dollar terms>",
    "insight": "<ONE sentence: the non-obvious second-order effect{'s of these stacked shocks' if is_compound else ''} that mainstream coverage misses>",
    "recommendation": "<Concrete trading action: hedge, hold, exit, or increase — with specific reasoning>"
}}"""

    text = await _call_haiku(prompt, max_tokens=700)

    if text:
        result = _extract_json(text)
        if result and all(k in result for k in ["impact", "risk", "insight"]):
            result["compound_mode"] = is_compound
            result["active_shock_count"] = combined_state.get("count", 1) if combined_state else 1
            return result
        else:
            logger.warning("Could not parse Trader's Brief from Haiku: %s", (text or "")[:300])

    result = _mock_traders_brief(shock_data, current_spread, new_spread, pnl_impact, position_mt, is_compound)
    result["active_shock_count"] = combined_state.get("count", 1) if combined_state else 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Fallback mock functions
# ═══════════════════════════════════════════════════════════════════════════
def _mock_headline_analysis(headline: str) -> Dict:
    """Fallback mock analysis when API is not available."""
    headline_lower = headline.lower()

    multiplier = 1.0
    freight = 0.0
    severity = "low"
    duration = 24.0

    if any(w in headline_lower for w in ["blockage", "blocked", "closure", "grounding"]):
        multiplier, freight, severity, duration = 1.6, 25.0, "high", 72
    elif any(w in headline_lower for w in ["ban", "embargo", "restrict", "sanction"]):
        multiplier, freight, severity, duration = 1.8, 5.0, "critical", 168
    elif any(w in headline_lower for w in ["attack", "missile", "drone", "strike"]):
        multiplier, freight, severity, duration = 1.7, 20.0, "high", 96
    elif any(w in headline_lower for w in ["storm", "typhoon", "flood", "weather"]):
        multiplier, freight, severity, duration = 1.3, 10.0, "medium", 48
    elif any(w in headline_lower for w in ["biodiesel", "mandate", "b50", "b40"]):
        multiplier, freight, severity, duration = 1.4, 0.0, "high", 336
    elif any(w in headline_lower for w in ["tariff", "duty", "tax", "levy"]):
        multiplier, freight, severity, duration = 1.2, 0.0, "medium", 168
    elif any(w in headline_lower for w in ["war", "conflict", "military", "naval"]):
        multiplier, freight, severity, duration = 2.0, 30.0, "critical", 120
    elif any(w in headline_lower for w in ["port", "congestion", "labor"]):
        multiplier, freight, severity, duration = 1.15, 8.0, "medium", 48

    return {
        "multiplier": multiplier,
        "freight_impact": freight,
        "severity": severity,
        "reasoning": (
            f"[Simulated] The headline suggests a {severity}-severity event "
            f"with a {multiplier:.1f}x basis volatility multiplier. "
            f"Configure ANTHROPIC_API_KEY in .env for real AI analysis."
        ),
        "duration_estimate_hours": duration,
        "simulated": True,
    }


def _mock_traders_brief(
    shock_data: Dict,
    current_spread: float,
    new_spread: float,
    pnl_impact: float,
    position_mt: float,
    is_compound: bool = False,
) -> Dict:
    spread_change = new_spread - current_spread
    pct_change = (spread_change / current_spread) * 100 if current_spread else 0
    event_label = shock_data.get("name", "event")
    compound_note = " (compound shock — multiple concurrent disruptions)" if is_compound else ""

    return {
        "impact": (
            f"{'Compounded shocks are' if is_compound else 'Shock is'} pushing the spread "
            f"by ${spread_change:.2f}/MT ({pct_change:+.1f}%) — "
            f"from ${current_spread:.2f} to ${new_spread:.2f}/MT{compound_note}. "
            f"Freight component adds ${shock_data.get('freight_impact', 0):.2f}/MT."
        ),
        "risk": (
            f"For a {position_mt:,.0f} MT long-spread position: "
            f"Base case P&L is ${pnl_impact:+,.0f}. "
            f"Worst case (shocks persist 2x): ${pnl_impact * 1.5:+,.0f}. "
            f"Best case (rapid resolution): ${pnl_impact * 0.4:+,.0f}."
        ),
        "insight": (
            f"[Simulated] {'Stacked shocks create non-linear stress — the compounded multiplier exceeds individual shock estimates and the market may not fully price the cascade effect.' if is_compound else f'The {event_label} creates a second-order squeeze on vessel availability that mainstream coverage underestimates — expect lagged freight repricing.'}"
        ),
        "recommendation": (
            f"{'With multiple shocks active, consider reducing position size until one resolves — correlation risk is elevated.' if is_compound else 'Hold current long-spread position if P&L is positive. Consider partial hedge via freight futures if exposure exceeds risk limits.'}"
        ),
        "simulated": True,
        "compound_mode": is_compound,
        "active_shock_count": 1,
    }
