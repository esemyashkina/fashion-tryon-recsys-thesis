"""Validate the packaged dataset and compute metrics from prediction snapshots."""

import argparse
import hashlib
import json
from math import log2
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(split):
    folder = ROOT / "data" / split
    cases = {row["id"]: row for row in read(folder / "scenarios.json")}
    people = {row["id"]: row for row in read(folder / "people.json")}
    catalog = {row["sku"]: row for row in read(folder / "catalog.json")}
    return folder, cases, people, catalog


def verify():
    image_sets = []
    for split in ("validation", "test"):
        folder, cases, people, catalog = load_split(split)
        hashes = set()
        for row in list(people.values()) + list(catalog.values()):
            digest = hashlib.sha256((folder / row["image_path"]).read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Image checksum differs: {row['image_path']}")
            hashes.add(digest)
        image_sets.append(hashes)
        for case in cases.values():
            pool = {sku for sku, item in catalog.items()
                    if all(item.get(field) == requested for field, requested in case["filters"].items())}
            if (case["photo_id"] not in people or pool != set(case["candidate_skus"])
                    or pool != set(case["review_options"].values())):
                raise ValueError(f"Invalid catalog pool or photo for {case['id']}")
    if image_sets[0] & image_sets[1]:
        raise ValueError("The splits share source images.")


def relevant(case):
    return {case["review_options"][n] for n in case["suitable_numbers"].replace(",", " ").split()}


def retrieval(rankings, gold, k):
    hits, recalls, reciprocal, ndcg = [], [], [], []
    for uid, rank in rankings.items():
        labels = gold[uid]
        hits.append(bool(labels.intersection(rank[:k])))
        recalls.append(len(labels.intersection(rank[:k])) / len(labels))
        first = next((i for i, sku in enumerate(rank, 1) if sku in labels), None)
        reciprocal.append(1 / first if first else 0)
        dcg = sum(1 / log2(i + 2) for i, sku in enumerate(rank[:k]) if sku in labels)
        ideal = sum(1 / log2(i + 2) for i in range(min(k, len(labels))))
        ndcg.append(dcg / ideal)
    return {"hit_count": sum(hits), "recall": mean(recalls), "mrr": mean(reciprocal), "ndcg": mean(ndcg)}


def summarize(cases, predictions):
    gold = {uid: relevant(case) for uid, case in cases.items()}
    rankings = predictions["rankings"]
    if rankings.keys() != cases.keys() or {uid: set(skus) for uid, skus in predictions["relevant_skus"].items()} != gold:
        raise ValueError("Predictions and current annotations differ.")
    for uid, row in rankings.items():
        pool = set(cases[uid]["candidate_skus"])
        for method in ("filter_text_ranking", "photo_ranking"):
            if set(row[method]) != pool or len(row[method]) != len(pool):
                raise ValueError(f"Invalid ranking {uid}/{method}")
    for name, choices in predictions["decisions"].items():
        if choices.keys() != cases.keys():
            raise ValueError(f"Incomplete predictions for {name}")
        if any(pick["sku"] not in rankings[uid]["photo_ranking"][:predictions["k"]]
               for uid, pick in choices.items()):
            raise ValueError(f"Selection outside shortlist for {name}")
    results = {}
    for group, ids in (
        ("all", list(cases)),
        ("original", [uid for uid, case in cases.items() if case["group"] == "review"]),
        ("partial_filters", [uid for uid, case in cases.items() if case["group"] == "relaxed_requests"]),
    ):
        results[group] = {
            "cases": len(ids),
            "photos": len({cases[uid]["photo_id"] for uid in ids}),
            "expected_random_top1": mean(len(gold[uid]) / len(cases[uid]["candidate_skus"]) for uid in ids),
            "retrieval": {
                method: {k: retrieval({uid: rankings[uid][method] for uid in ids}, gold, k) for k in (1, 2, 4, 6)}
                for method in ("filter_text_ranking", "photo_ranking")
            },
            "prototype_top1": {
                name: sum(choices[uid]["sku"] in gold[uid] for uid in ids)
                for name, choices in predictions["decisions"].items()
            },
        }
    return results


def summarize_k(cases, predictions):
    """Recompute the validation comparison at each shortlist size."""
    gold = {uid: relevant(case) for uid, case in cases.items()}
    rankings = predictions["rankings"]
    results = {}
    for k, decisions in predictions["k_trials"].items():
        size = int(k)
        shortlists = {uid: row["photo_ranking"][:size] for uid, row in rankings.items()}
        if decisions.keys() != cases.keys() or any(sku not in shortlists[uid] for uid, sku in decisions.items()):
            raise ValueError(f"Invalid shortlist decisions for K={size}")
        results[k] = {
            "top1_count": sum(sku in gold[uid] for uid, sku in decisions.items()),
            "hit_count": sum(bool(gold[uid].intersection(skus)) for uid, skus in shortlists.items()),
            "cases": len(cases),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "metrics.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    verify()
    result = {}
    for split in ("validation", "test"):
        cases = load_split(split)[1]
        predictions = read(args.predictions / f"{split}_predictions.json")
        result[split] = summarize(cases, predictions)
        if "k_trials" in predictions:
            result[split]["k_comparison"] = summarize_k(cases, predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
