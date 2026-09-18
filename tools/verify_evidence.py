"""Recompute the public rate-study aggregate using only the Python standard library.

This checks arithmetic and internal consistency, not the unavailable raw predictions.
An optional original report additionally authenticates the exported projection.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "922679ce9b665c836cd7f21b8af6aa5de0d0c23545da6e7d69ff30bb7e8e2943"
ARMS = ("control", "lower")
ENDPOINTS = ("0", "648", "1296", "2592")
FAMILIES = ("color", "count", "switch")
BROAD = ("dev", "retention", "train_fit", "transfer_original", "transfer_varied")
BASIS = tuple("basis_" + panel for panel in ("binding", "revision", "composition", "sequence"))
DEFINITION = tuple("definition_" + panel for panel in ("binding", "revision", "composition"))
ACQUISITION = tuple(
    "complementary_" + split + "_" + panel
    for split in ("fit", "dev") for panel in ("binding", "revision")
)
BANKS = BASIS + ACQUISITION + DEFINITION + BROAD
RETENTION_BANKS = BASIS + DEFINITION + BROAD
BROAD_METRICS = ("anchor_pair_both", "known_action", "known_reply", "unknown_action", "unknown_reply")
REFERENCE_FIELDS = {
    "prior13000": "prior13000_parent_metrics",
    "previous10408": "previous10408_curriculum_metrics",
    "original9760": "original9760_parent_metrics",
    "historical9112": "historical9112_reference_metrics",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metric_names(bank: str) -> tuple[str, ...]:
    return BROAD_METRICS if bank in BROAD else ("all_query_pair_both",)


def project_metrics(metrics: dict, banks: tuple[str, ...]) -> dict:
    """Keep counts and denominators; discard rounded rates and unrelated metrics."""
    def counts(group: dict, bank: str) -> dict:
        return {
            name: [group["counts"][name]["count"], group["counts"][name]["total"]]
            for name in metric_names(bank)
        }

    return {
        bank: {
            "overall": counts(metrics[bank]["overall"], bank),
            "families": {
                family: counts(metrics[bank]["by_family"][family], bank)
                for family in FAMILIES
            },
        }
        for bank in banks
    }


def project_report(report: dict) -> dict:
    """A deterministic projection, not an independent raw-output recount."""
    require(report["status"] == "verified", "source report is not verified")
    require(
        report["common15592_parent_metrics"]["control"]
        == report["common15592_parent_metrics"]["lower"]
        == report["native_metrics"]["control"]["0"]
        == report["native_metrics"]["lower"]["0"],
        "source does not have the common baseline",
    )
    screen = report["screen"]
    reported_arms = {}
    for arm in ARMS:
        outcome = screen["arms"][arm]
        reported_arms[arm] = {
            "acquisition_passes": sum(outcome["acquisition_checks"].values()),
            "capability_passes": sum(outcome["capability_checks"].values()),
            "retention_passes": {
                reference: sum(result["checks"].values())
                for reference, result in outcome["retention"].items()
            },
            "original_transfer_mean_pp": outcome["transfer_gains"]["original9760"]["equal_cell_mean_pp"],
            "eligible": outcome["eligible"],
        }
    return {
        "schema": "bic-public-rate-aggregate-v1",
        "provenance": {
            "kind": "derived_projection",
            "source_report": "runs/shared-rate-analysis-local/attempt-002/independent-analysis.json",
            "source_report_sha256": SOURCE_SHA256,
            "source_status": report["status"],
            "launch_sha256": report["launch_sha256"],
            "summary_sha256": report["summary_sha256"],
            "limitations": "Aggregate arithmetic only. Raw episode predictions, full optimizer archives and the complete historical input closure are not included.",
        },
        "design": {
            "common_lifetime_updates": 15592,
            "additional_updates": [int(step) for step in ENDPOINTS],
            "learning_rates": {"control": 0.0003, "lower": 0.0001},
            "auxiliary_weight": 0.3,
            "acquisition_minimum_fraction": [3, 4],
            "capability_minimum_fraction": [3, 4],
            "maximum_retention_loss_fraction": [1, 20],
            "effect_minimum_fraction": [1, 20],
            "equal_weight_unit": "bank/family cell",
        },
        "observations": {
            arm: {
                endpoint: project_metrics(report["native_metrics"][arm][endpoint], BANKS)
                for endpoint in ENDPOINTS
            }
            for arm in ARMS
        },
        "references": {
            name: project_metrics(report[field], RETENTION_BANKS)
            for name, field in REFERENCE_FIELDS.items()
        },
        "reported": {
            "acquisition_mean_effect_pp": screen["lower_minus_control_equal_cell_mean_pp"],
            "split_mean_effect_pp": screen["split_mean_effect_pp"],
            "family_mean_effect_pp": screen["family_mean_effect_pp"],
            "rate_effect_passed": screen["rate_effect_passed"],
            "secondary_mean_effect_pp": screen["secondary_basis_transfer_mean_pp"],
            "secondary_effect_passed": screen["secondary_basis_transfer_passed"],
            "arms": reported_arms,
            "eligible_arms": screen["eligible_arms"],
            "automatic_adoption": screen["automatic_adoption"],
            "selected_arm": screen["selected_arm"],
        },
    }


def count_pair(value: list) -> tuple[int, int]:
    require(isinstance(value, list) and len(value) == 2, "count must be [successes, total]")
    numerator, denominator = value
    require(
        type(numerator) is int and type(denominator) is int
        and denominator > 0 and 0 <= numerator <= denominator,
        "invalid integer count or denominator",
    )
    return numerator, denominator


def validate_metrics(metrics: dict, banks: tuple[str, ...]) -> None:
    require(set(metrics) == set(banks), "bank inventory differs")
    for bank in banks:
        data = metrics[bank]
        require(set(data) == {"overall", "families"}, "bank fields differ")
        require(set(data["families"]) == set(FAMILIES), "family inventory differs")
        groups = [data["overall"], *data["families"].values()]
        for group in groups:
            require(set(group) == set(metric_names(bank)), "metric inventory differs")
            for value in group.values():
                count_pair(value)
        for metric in metric_names(bank):
            family_counts = [count_pair(data["families"][f][metric]) for f in FAMILIES]
            require(
                count_pair(data["overall"][metric])
                == tuple(sum(item[index] for item in family_counts) for index in (0, 1)),
                "family totals do not add to the overall metric",
            )


def rate(metrics: dict, bank: str, family: str, metric: str = "all_query_pair_both") -> Fraction:
    return Fraction(*count_pair(metrics[bank]["families"][family][metric]))


def delta(after: dict, before: dict, bank: str, family: str, metric: str = "all_query_pair_both") -> Fraction:
    new = count_pair(after[bank]["families"][family][metric])
    old = count_pair(before[bank]["families"][family][metric])
    require(new[1] == old[1], "compared denominators differ")
    return Fraction(new[0] - old[0], new[1])


def mean(values: list[Fraction]) -> Fraction:
    require(bool(values), "cannot average an empty cell set")
    return sum(values, Fraction()) / len(values)


def transfer(after: dict, before: dict) -> list[Fraction]:
    return [delta(after, before, bank, family)
            for bank in ("basis_composition", "basis_sequence") for family in FAMILIES]


def close(actual: Fraction, expected: float, label: str) -> None:
    require(abs(float(actual) - expected) <= 1e-10, label + " differs from reported value")


def verify(data: dict) -> dict:
    require(data["schema"] == "bic-public-rate-aggregate-v1", "unsupported evidence schema")
    require(data["provenance"]["source_report_sha256"] == SOURCE_SHA256, "source pin differs")
    require(data["provenance"]["kind"] == "derived_projection", "projection label missing")
    design = data["design"]
    require(design["additional_updates"] == [int(n) for n in ENDPOINTS], "endpoint schedule differs")
    require(design["common_lifetime_updates"] == 15592, "common checkpoint differs")
    require(design["learning_rates"] == {"control": 0.0003, "lower": 0.0001}, "rate contrast differs")
    require(design["auxiliary_weight"] == 0.3, "auxiliary objective differs")
    for key, expected in (
        ("acquisition_minimum_fraction", [3, 4]),
        ("capability_minimum_fraction", [3, 4]),
        ("maximum_retention_loss_fraction", [1, 20]),
        ("effect_minimum_fraction", [1, 20]),
    ):
        require(design[key] == expected, "prospective threshold differs: " + key)
    observations = data["observations"]
    require(set(observations) == set(ARMS), "arm inventory differs")
    for arm in ARMS:
        require(set(observations[arm]) == set(ENDPOINTS), "endpoint inventory differs")
        for endpoint in ENDPOINTS:
            validate_metrics(observations[arm][endpoint], BANKS)
    baseline = observations["control"]["0"]
    require(baseline == observations["lower"]["0"], "starting metrics differ between arms")
    require(set(data["references"]) == set(REFERENCE_FIELDS), "historical reference inventory differs")
    for reference in data["references"].values():
        validate_metrics(reference, RETENTION_BANKS)
    # Stable denominators at every endpoint prevent incomparable trajectories.
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            for bank in BANKS:
                for family in FAMILIES:
                    for metric in metric_names(bank):
                        delta(observations[arm][endpoint], baseline, bank, family, metric)
    final = {arm: observations[arm][ENDPOINTS[-1]] for arm in ARMS}
    effects = {(bank, family): delta(final["lower"], final["control"], bank, family)
               for bank in ACQUISITION for family in FAMILIES}
    split_means = {split: mean([value for (bank, _), value in effects.items() if "_" + split + "_" in bank])
                   for split in ("fit", "dev")}
    family_means = {family: mean([value for (_, name), value in effects.items() if name == family])
                    for family in FAMILIES}
    acquisition_mean = mean(list(effects.values()))
    minimum_effect = Fraction(*design["effect_minimum_fraction"])
    primary_pass = (acquisition_mean >= minimum_effect and all(value >= 0 for value in effects.values())
                    and all(value > 0 for value in split_means.values())
                    and all(value > 0 for value in family_means.values()))
    secondary_cells = transfer(final["lower"], final["control"])
    secondary_mean = mean(secondary_cells)
    secondary_pass = secondary_mean >= minimum_effect and all(value >= 0 for value in secondary_cells)
    reported = data["reported"]
    close(acquisition_mean * 100, reported["acquisition_mean_effect_pp"], "acquisition mean")
    for split, value in split_means.items():
        close(value * 100, reported["split_mean_effect_pp"][split], split + " mean")
    for family, value in family_means.items():
        close(value * 100, reported["family_mean_effect_pp"][family], family + " mean")
    close(secondary_mean * 100, reported["secondary_mean_effect_pp"], "secondary mean")
    require(primary_pass == reported["rate_effect_passed"], "primary screen differs")
    require(secondary_pass == reported["secondary_effect_passed"], "secondary screen differs")
    references = {"common15592": baseline, **data["references"]}
    results = {}
    for arm in ARMS:
        acquisition = [rate(final[arm], bank, family) >= Fraction(*design["acquisition_minimum_fraction"])
                       for bank in ACQUISITION for family in FAMILIES]
        capability = [rate(final[arm], bank, family) >= Fraction(*design["capability_minimum_fraction"])
                      for bank in BASIS for family in FAMILIES]
        retained, failures = {}, []
        for label, reference in references.items():
            checks = []
            for bank in RETENTION_BANKS:
                for family in FAMILIES:
                    for metric in metric_names(bank):
                        difference = delta(final[arm], reference, bank, family, metric)
                        passed = difference >= -Fraction(*design["maximum_retention_loss_fraction"])
                        checks.append(passed)
                        if not passed:
                            failures.append({"reference": label, "bank": bank, "family": family,
                                             "metric": metric, "change_pp": float(difference * 100)})
            require(len(checks) == 96, "each retention reference requires 96 checks")
            retained[label] = sum(checks)
        original_gain = mean(transfer(final[arm], references["original9760"]))
        eligible = all(acquisition) and all(capability) and original_gain > 0 and not failures
        result = {"acquisition_passes": sum(acquisition), "capability_passes": sum(capability),
                  "retention_passes": retained, "original_transfer_mean_pp": float(original_gain * 100),
                  "eligible": eligible}
        require(result == reported["arms"][arm], arm + " outcome differs from reported values")
        results[arm] = {**result, "total_retention_passes": sum(retained.values()),
                        "total_retention_checks": 96 * len(references), "retention_failures": failures}
    eligible_arms = [arm for arm in ARMS if results[arm]["eligible"]]
    require(eligible_arms == reported["eligible_arms"], "eligible arm list differs")
    require(reported["automatic_adoption"] is False and reported["selected_arm"] is None,
            "this aggregate has no automatic selection or adoption")
    return {"status": "aggregate arithmetic verified", "bank_trajectories": len(BANKS),
            "acquisition_mean_effect_pp": float(acquisition_mean * 100),
            "rate_effect_passed": primary_pass, "secondary_mean_effect_pp": float(secondary_mean * 100),
            "secondary_effect_passed": secondary_pass, "arms": results, "eligible_arms": eligible_arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "shared-rate-aggregate.json")
    parser.add_argument("--source-report", type=Path, help="Optional original report; checks its pinned SHA and exact projection")
    args = parser.parse_args()
    payload = args.evidence.read_bytes()
    data = json.loads(payload)
    result = verify(data)
    result["evidence_sha256"] = hashlib.sha256(payload).hexdigest()
    result["original_report_authenticated"] = False
    if args.source_report is not None:
        original = args.source_report.read_bytes()
        require(hashlib.sha256(original).hexdigest() == SOURCE_SHA256, "original report SHA-256 differs")
        require(project_report(json.loads(original)) == data, "projection differs from original report")
        result["original_report_authenticated"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
