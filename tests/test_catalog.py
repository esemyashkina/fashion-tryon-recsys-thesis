from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import catalog
from src.evaluate import load_split


class CatalogTest(unittest.TestCase):
    def test_sql_filters_match_all_validation_pools(self):
        _, cases, _, items = load_split("validation")
        with TemporaryDirectory() as temp:
            db = catalog.catalog_db(output_dir=Path(temp))
            with patch.object(catalog, "catalog_db", return_value=db):
                for uid, case in cases.items():
                    self.assertEqual(set(catalog.candidate_skus(case["filters"])), set(case["candidate_skus"]), uid)
                for category in {item["category"] for item in items.values()}:
                    self.assertEqual(set(catalog.candidate_skus({"category": category})),
                                     {sku for sku, item in items.items() if item["category"] == category})
