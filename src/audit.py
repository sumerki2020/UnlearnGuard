"""Stage 6 CLI: aggregate per-seed tier results into Deletion Durability Cards.

Reads results/tiers_<method>_seed<seed>.json, builds one card per method, writes
JSON + Markdown to the cards dir, and (in strict mode) exits nonzero if any
required tier is FAIL or INCONCLUSIVE. PROVISIONAL (single-seed, all-PASS) exits
zero but is labeled.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

from src import card as cardlib
from src.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true", help="override cfg.strict = true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    strict = args.strict or cfg.get("strict", False)
    results_dir = Path(cfg.get("results_dir", "results"))
    cards_dir = Path(cfg.get("cards_dir", "results/cards"))
    cards_dir.mkdir(parents=True, exist_ok=True)

    failing = []
    for method in cfg["methods"]:
        files = sorted(glob.glob(str(results_dir / f"tiers_{method}_seed*.json")))
        if not files:
            print(f"[{method}] no tier results found — skipping")
            continue
        seed_results, model, provenance = [], cfg["model_name"], {}
        for f in files:
            payload = json.load(open(f))
            data = payload["data"] if "data" in payload else payload
            seed_results.append(data)
            m = data.get("manifest", {})
            provenance = {
                "config_hash": m.get("config_hash"), "dataset_hash": m.get("dataset_hash"),
                "container_image_digest": m.get("container_image_digest"),
                "git_commit": m.get("git_commit", "unknown"),
                "model_version": model, "judge_version": None,
                "hardware": m.get("hardware"), "runtime_seconds": m.get("runtime_seconds"),
                "estimated_cost_usd": None,
            }

        c = cardlib.build_card(method, model, seed_results, cfg, provenance)
        cardlib.validate(c, cfg["schema"])
        json_path = cards_dir / f"card_{method}.json"
        md_path = cards_dir / f"card_{method}.md"
        json.dump(c, open(json_path, "w"), indent=2, sort_keys=True)
        open(md_path, "w").write(cardlib.to_markdown(c))
        print(f"[{method}] overall={c['overall_status']}  -> {json_path}, {md_path}")
        for t in c["tiers"]:
            print(f"    tier {t['tier']} {t['name']:32s} {t['status']}")
        if c["overall_status"] in ("FAIL", "INCONCLUSIVE"):
            failing.append((method, c["overall_status"]))

    if strict and failing:
        print(f"\nSTRICT: nonzero exit — {failing}")
        sys.exit(2)
    print("\naudit complete" + (" (strict OK)" if strict else ""))


if __name__ == "__main__":
    main()
