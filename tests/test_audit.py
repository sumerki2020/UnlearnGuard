"""Deterministic Stage 7 fixtures: exercise the stats -> card -> status path
without training. A known-PASS scenario must yield PASS; a known-FAIL scenario
must yield FAIL. Run directly: `python tests/test_audit.py` (exit 0 on success).
"""

import json
from pathlib import Path

from src import card as cardlib

CFG = {"card_version": "test", "operating_point": 16, "n_boot": 500,
       "model_name": "fixture"}
SCHEMA = str(Path(__file__).resolve().parents[1] / "schemas"
             / "deletion-durability-card.schema.json")
CANARIES = ["verbatim_0", "phone_0", "shared_0"]
SEEDS = [1, 2, 3]


def _seed_result(seed, unlearned_rate, gold_rate, gold_jitter):
    """One seed's tier results; gold_jitter gives the control its natural
    across-canary spread that sets the equivalence margin."""
    def block(base, jitter):
        return {c: max(0.0, base + (i - 1) * jitter) for i, c in enumerate(CANARIES)}
    tier = lambda: {"unlearned": block(unlearned_rate, gold_jitter),
                    "gold": block(gold_rate, gold_jitter)}
    return {"method": "ga", "seed": seed, "tier1": tier(), "tier2": tier(),
            "tier3": tier(), "utility": {}, "n_prompts_per_canary": 4,
            "manifest": {"git_commit": "test", "config_hash": None,
                         "dataset_hash": None, "hardware": "ci"}}


def _card(unlearned_rate, gold_rate, gold_jitter):
    seed_results = [_seed_result(s, unlearned_rate, gold_rate, gold_jitter) for s in SEEDS]
    c = cardlib.build_card("ga", "fixture", seed_results, CFG,
                           {"git_commit": "test", "config_hash": None,
                            "dataset_hash": None})
    cardlib.validate(c, SCHEMA)
    return c


def main():
    # KNOWN PASS: unlearned recovers no more than the gold control
    p = _card(unlearned_rate=0.02, gold_rate=0.05, gold_jitter=0.05)
    print("known-pass overall:", p["overall_status"],
          "| tier statuses:", [t["status"] for t in p["tiers"]])
    assert p["overall_status"] == "PASS", p["overall_status"]

    # KNOWN FAIL: unlearned still leaks strongly vs a chance-level control
    f = _card(unlearned_rate=0.95, gold_rate=0.02, gold_jitter=0.02)
    print("known-fail overall:", f["overall_status"],
          "| tier statuses:", [t["status"] for t in f["tiers"]])
    assert f["overall_status"] == "FAIL", f["overall_status"]

    print("\nCARD FIXTURES OK: PASS and FAIL scenarios resolve correctly")


if __name__ == "__main__":
    main()
