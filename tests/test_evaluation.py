import json
import unittest

from src.evaluate import ROOT, load_split, read, summarize, summarize_k, verify


class EvaluationTest(unittest.TestCase):
    def test_published_results(self):
        verify()
        recalculated = {}
        snapshots = {}
        for split, count in (("validation", 114), ("test", 57)):
            with self.subTest(split=split):
                snapshot = read(ROOT / "reports" / f"{split}_predictions.json")
                cases = load_split(split)[1]
                result = summarize(cases, snapshot)
                self.assertEqual(result["all"]["cases"], count)
                if split == "validation":
                    result["k_comparison"] = summarize_k(cases, snapshot)
                recalculated[split] = result
                snapshots[split] = snapshot
        self.assertEqual(json.loads(json.dumps(recalculated)), read(ROOT / "reports" / "metrics.json"))

        comparison = recalculated["validation"]["k_comparison"]
        best_k = max(comparison, key=lambda k: (comparison[k]["top1_count"], -int(k)))
        self.assertEqual(snapshots["validation"]["k"], int(best_k))
        self.assertEqual(snapshots["test"]["k"], int(best_k))
        roles = ("v3", "finished_looks", "style_rules")
        scores = recalculated["validation"]["all"]["prototype_top1"]
        winner = max(roles, key=scores.__getitem__)
        self.assertEqual(set(snapshots["test"]["decisions"]), {winner})
        self.assertEqual(snapshots["test"]["prompts"][winner], snapshots["validation"]["prompts"][winner])
