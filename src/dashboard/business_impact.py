"""Business-impact scenario calculations for the executive dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BusinessImpact:
    production_volume: int
    expected_failures: float
    expected_alerts: float
    expected_true_positive_alerts: float
    potentially_prevented_failures: float
    gross_avoided_failure_cost: float
    alert_review_cost: float
    net_estimated_impact: float
    estimated_roi: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def calculate_business_impact(
    production_volume: int,
    failure_rate: float,
    alert_rate: float,
    precision: float,
    intervention_effectiveness: float,
    cost_per_failure: float,
    cost_per_alert_review: float,
) -> BusinessImpact:
    """Calculate a transparent scenario; rates are supplied as decimals."""
    expected_failures = production_volume * failure_rate
    expected_alerts = production_volume * alert_rate
    expected_true_positive_alerts = min(
        expected_failures, expected_alerts * precision
    )
    potentially_prevented = (
        expected_true_positive_alerts * intervention_effectiveness
    )
    gross_avoided_cost = potentially_prevented * cost_per_failure
    review_cost = expected_alerts * cost_per_alert_review
    net_impact = gross_avoided_cost - review_cost
    roi = net_impact / review_cost if review_cost > 0 else None
    return BusinessImpact(
        production_volume=int(production_volume),
        expected_failures=expected_failures,
        expected_alerts=expected_alerts,
        expected_true_positive_alerts=expected_true_positive_alerts,
        potentially_prevented_failures=potentially_prevented,
        gross_avoided_failure_cost=gross_avoided_cost,
        alert_review_cost=review_cost,
        net_estimated_impact=net_impact,
        estimated_roi=roi,
    )
