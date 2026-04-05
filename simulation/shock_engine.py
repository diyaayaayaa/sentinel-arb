"""
What-If Shock Simulation Engine
================================
Three shock types:
  1. Logistics Shock: Port outage / Suez-style blockage
  2. Policy Shock: Export ban / biodiesel mandate
  3. Custom AI Shock: Claude Haiku interprets a news headline
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ShockProfile:
    """Defines the parameters of a market shock."""
    shock_type: str
    name: str
    freight_impact: float        # $/MT added to freight
    spread_multiplier: float     # Multiplier on basis spread
    duration_hours: float        # How long shock persists
    description: str
    severity: str                # "low", "medium", "high", "critical"


# Pre-configured shock scenarios
LOGISTICS_SCENARIOS = {
    "malacca_blockage": ShockProfile(
        shock_type="logistics",
        name="Malacca Strait Blockage",
        freight_impact=35.0,
        spread_multiplier=1.25,
        duration_hours=72,
        description=(
            "Major vessel grounding in the Malacca Strait disrupts "
            "60% of palm oil shipping routes to Europe. Freight rates "
            "surge as vessels reroute via Lombok Strait (+5 days transit)."
        ),
        severity="critical",
    ),
    "suez_blockage": ShockProfile(
        shock_type="logistics",
        name="Suez Canal Disruption",
        freight_impact=25.0,
        spread_multiplier=1.18,
        duration_hours=96,
        description=(
            "Suez Canal blocked, forcing Cape of Good Hope rerouting. "
            "Adds 10-14 days to Malaysia-Rotterdam transit. "
            "Freight costs spike; Rotterdam premium widens."
        ),
        severity="high",
    ),
    "port_klang_outage": ShockProfile(
        shock_type="logistics",
        name="Port Klang Outage",
        freight_impact=15.0,
        spread_multiplier=1.10,
        duration_hours=48,
        description=(
            "Major infrastructure failure at Port Klang, Malaysia's "
            "largest palm oil export terminal. Loading delays of 3-5 "
            "days cascade through supply chain."
        ),
        severity="medium",
    ),
    "rotterdam_congestion": ShockProfile(
        shock_type="logistics",
        name="Rotterdam Port Congestion",
        freight_impact=10.0,
        spread_multiplier=1.08,
        duration_hours=120,
        description=(
            "Severe congestion at Rotterdam port due to labor action. "
            "Demurrage costs rise; CIF premium expands as "
            "discharge delays mount."
        ),
        severity="medium",
    ),
}

POLICY_SCENARIOS = {
    "indonesia_export_ban": ShockProfile(
        shock_type="policy",
        name="Indonesia Export Ban",
        freight_impact=5.0,
        spread_multiplier=1.20,
        duration_hours=168,  # 1 week
        description=(
            "Indonesia imposes sudden palm oil export ban to stabilize "
            "domestic cooking oil prices. Malaysia becomes sole major "
            "supplier; FOB prices surge 15-20%."
        ),
        severity="critical",
    ),
    "b50_mandate": ShockProfile(
        shock_type="policy",
        name="Indonesia B50 Biodiesel Mandate",
        freight_impact=0.0,
        spread_multiplier=1.15,
        duration_hours=720,  # 30 days
        description=(
            "Indonesia accelerates B50 biodiesel mandate, diverting "
            "8-10 million MT of CPO to domestic fuel blending. "
            "Global supply tightens; basis widens structurally."
        ),
        severity="high",
    ),
    "eu_deforestation_reg": ShockProfile(
        shock_type="policy",
        name="EU Deforestation Regulation (EUDR)",
        freight_impact=8.0,
        spread_multiplier=1.12,
        duration_hours=720,
        description=(
            "Strict EUDR enforcement blocks non-compliant palm oil "
            "shipments at Rotterdam. Compliant supply premium emerges; "
            "certified sustainable palm oil trades at $40-60/MT premium."
        ),
        severity="high",
    ),
    "india_import_duty": ShockProfile(
        shock_type="policy",
        name="India Import Duty Hike",
        freight_impact=0.0,
        spread_multiplier=0.92,
        duration_hours=336,  # 14 days
        description=(
            "India raises import duties on refined palm oil by 10%. "
            "Demand redirects to crude palm oil; Rotterdam refined "
            "premium narrows as European demand softens."
        ),
        severity="medium",
    ),
}


def get_logistics_shock(scenario_key: str) -> Optional[Dict]:
    """Get a pre-configured logistics shock."""
    profile = LOGISTICS_SCENARIOS.get(scenario_key)
    if not profile:
        return None
    return _profile_to_dict(profile)


def get_policy_shock(scenario_key: str) -> Optional[Dict]:
    """Get a pre-configured policy shock."""
    profile = POLICY_SCENARIOS.get(scenario_key)
    if not profile:
        return None
    return _profile_to_dict(profile)


def build_custom_shock(
    freight_impact: float,
    spread_multiplier: float,
    duration_hours: float,
    description: str,
    severity: str = "medium",
) -> Dict:
    """Build a custom shock from AI analysis or manual input."""
    return {
        "type": "ai_custom",
        "name": "Custom AI Shock",
        "freight_impact": freight_impact,
        "spread_multiplier": spread_multiplier,
        "duration_hours": duration_hours,
        "description": description,
        "severity": severity,
    }


def list_scenarios() -> Dict:
    """Return all available pre-configured scenarios."""
    return {
        "logistics": {
            k: {
                "name": v.name,
                "description": v.description,
                "severity": v.severity,
                "freight_impact": v.freight_impact,
                "spread_multiplier": v.spread_multiplier,
                "duration_hours": v.duration_hours,
            }
            for k, v in LOGISTICS_SCENARIOS.items()
        },
        "policy": {
            k: {
                "name": v.name,
                "description": v.description,
                "severity": v.severity,
                "freight_impact": v.freight_impact,
                "spread_multiplier": v.spread_multiplier,
                "duration_hours": v.duration_hours,
            }
            for k, v in POLICY_SCENARIOS.items()
        },
    }


def _profile_to_dict(profile: ShockProfile) -> Dict:
    return {
        "type": profile.shock_type,
        "name": profile.name,
        "freight_impact": profile.freight_impact,
        "spread_multiplier": profile.spread_multiplier,
        "duration_hours": profile.duration_hours,
        "description": profile.description,
        "severity": profile.severity,
    }
