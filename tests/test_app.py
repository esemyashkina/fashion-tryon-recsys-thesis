import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Install the application dependencies")
class AppTest(unittest.TestCase):
    def test_gradio_uses_the_validation_catalog_for_both_workflows(self):
        from PIL import Image
        from src import app, tryon

        self.assertEqual((len(app.PEOPLE), len(app.CATALOG)), (52, 59))
        self.assertTrue(all(uid.startswith("validation:") for uid in app.PEOPLE | app.CATALOG))
        snapshot = app.read(app.ROOT / "reports/test_predictions.json")
        for field in ("model", "reasoning_effort", "k"):
            self.assertEqual(app.CONFIG[field], snapshot[field])
        self.assertEqual(app.SELECTION_PROMPT, snapshot["prompts"][app.CONFIG["selection_prompt"]])
        demo = app.build_app()
        try:
            endpoints = {dependency["api_name"] for dependency in demo.config["dependencies"]}
            self.assertTrue({"try_direct", "run_recommendation", "try_recommended"} <= endpoints)
        finally:
            demo.close()

        person_id = next(iter(app.PEOPLE))
        _, sku = app.product_options("tops")[0]
        expected = Image.new("RGB", (4, 4))
        calls = []

        def backend(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[expected])

        with patch.object(tryon, "load_pipeline", return_value=backend):
            output = app.try_direct(None, person_id, None, sku, "tops", "model", 10)
            photo = app.person_photo(None, person_id)
            self.assertIs(app.try_recommended(photo, sku, 10), expected)
        self.assertIs(output, expected)
        self.assertEqual([call["garment_photo_type"] for call in calls], ["flat-lay", "flat-lay"])
        self.assertEqual([call["num_timesteps"] for call in calls], [10, 10])
