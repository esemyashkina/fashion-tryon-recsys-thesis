"""Run FashionCLIP retrieval and optionally request fresh Terra decisions."""

import argparse
import json
from functools import partial

from .evaluate import ROOT, load_split, read, relevant
from .recommendation import Terra, catalog_index, load_encoder, rank
from .tryon import open_photo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("validation", "test"))
    parser.add_argument("--live", action="store_true", help="Generate new queries and decisions with the paid API.")
    parser.add_argument("--only", help="Run a single final-selection prompt with --live.")
    parser.add_argument("--sweep-k", action="store_true", help="Also evaluate K=2, 4, 6 with v3 on validation.")
    args = parser.parse_args()
    if args.only and not args.live:
        parser.error("--only requires --live.")
    if args.sweep_k and (not args.live or args.split != "validation"):
        parser.error("--sweep-k requires a live validation run.")
    folder, cases, people, catalog = load_split(args.split)
    snapshot = read(ROOT / "reports" / f"{args.split}_predictions.json")
    prompts = snapshot["prompts"]
    if args.only:
        if args.only not in prompts:
            parser.error(f"Available prompts: {', '.join(prompts)}")
        prompts = {args.only: prompts[args.only]}
    terra = Terra(snapshot["model"], snapshot["reasoning_effort"]) if args.live else None
    encoder = load_encoder()
    vectors = catalog_index(encoder, folder, catalog)
    rank_pool = partial(rank, encoder, vectors)

    def open_product(sku):
        return open_photo(folder / catalog[sku]["image_path"])

    ranks = {}
    decisions = {name: {} for name in prompts}
    k_trials = {k: {} for k in (2, 4, 6)} if args.sweep_k else {}
    output = ROOT / "runs" / (f"{args.split}_predictions.json" if args.live else f"{args.split}_rankings.json")
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Move the previous result before running: {output}")
    for uid, case in cases.items():
        filters = case["filters"]
        pool = case["candidate_skus"]
        label = " ".join(filters[f] for f in ("color", "style", "garment_type") if f in filters).replace("_", " ")
        label = label or {"tops": "top", "bottoms": "bottoms", "one-pieces": "one-piece garment"}[filters["category"]]
        if args.live:
            photo = open_photo(folder / people[case["photo_id"]]["image_path"])
            query, context = terra.query(photo, filters)
        else:
            query = snapshot["rankings"][uid]["query"]
        photo_ranking = rank_pool(query["query"], pool)
        ranks[uid] = {"filter_text_ranking": rank_pool(label, pool), "photo_ranking": photo_ranking, "query": query}
        if args.live:
            choices = {}

            def choose(prompt, shortlist):
                key = prompt, tuple(shortlist)
                if key in choices:
                    return choices[key]
                choice = terra.choose(prompt, context, shortlist, open_product)
                choices[key] = choice
                return choice

            for name, prompt in prompts.items():
                decisions[name][uid] = choose(prompt, photo_ranking[:snapshot["k"]])
            for k, picks in k_trials.items():
                picks[uid] = choose(snapshot["prompts"]["v3"], photo_ranking[:k])["sku"]
            print(uid, flush=True)
    if args.live:
        result = {"rankings": ranks, "decisions": decisions, "k": snapshot["k"],
                  "model": snapshot["model"], "reasoning_effort": snapshot["reasoning_effort"], "prompts": prompts,
                  "relevant_skus": {uid: sorted(relevant(case)) for uid, case in cases.items()}}
        if k_trials:
            result["k_trials"] = k_trials
    else:
        changed = sum(row != snapshot["rankings"][uid] for uid, row in ranks.items())
        print(f"{len(ranks) - changed}/{len(ranks)} full rankings match the saved snapshot.")
        result = ranks
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
