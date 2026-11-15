"""Queue/bridge regression checks; no model is loaded or generated."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi import HTTPException

import direct_server as server
import minimax_bridge


class MiniMaxQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patchers = [patch.object(server, "OUTPUT", root / "output"),
                         patch.object(server, "INPUT", root / "input"),
                         patch.object(server, "save_settings"),
                         patch.object(server.torch.cuda, "is_available", return_value=True),
                         patch.object(minimax_bridge, "status", return_value={"models": {"image": True, "video": True}})]
        for p in self.patchers:
            p.start()
        server.OUTPUT.mkdir(); server.INPUT.mkdir()
        Image.new("RGB", (512, 512), "orange").save(server.INPUT / "ref.png")
        server.STATE_LOCK.acquire()  # prevent the background worker from claiming synthetic jobs
        self.before = list(server.QUEUE)
        self.jobs_before = list(server.JOBS)
        server.QUEUE.clear(); server.JOBS.clear()

    def tearDown(self):
        server.QUEUE[:] = self.before
        server.JOBS[:] = self.jobs_before
        server.STATE_LOCK.release()
        for p in reversed(self.patchers):
            p.stop()
        self.temp.cleanup()

    def test_sheet_presets_and_h3_recipe(self):
        result = server.queue({"project": "QA", "mode": "sheet", "sheet": "turnaround5", "bg": "grey",
                               "models": ["minimax-h3-image"], "refs": ["ref.png"], "count": 1,
                               "steps": 20, "cfg": 7, "size": "wide"})
        self.assertEqual(result["queued"], 1)
        job = server.QUEUE[0]
        self.assertIn("five-view", job["prompt"])
        self.assertIn("<Picture 1>", job["prompt"])
        self.assertIn("light grey", job["prompt"])
        self.assertEqual((job["steps"], job["cfg"]), (6, 1.0))
        self.assertEqual(job["width"] % 32, 0)
        self.assertEqual(job["height"] % 32, 0)

    def test_h3_text_only_and_qwen_reference_requirement(self):
        body = {"project": "QA", "mode": "free", "models": ["minimax-h3-image"], "raw": "A red panda.", "count": 1}
        self.assertEqual(server.queue(body)["queued"], 1)
        self.assertEqual(server.QUEUE[-1]["refs"], [])
        with self.assertRaises(HTTPException):
            server.queue({**body, "models": ["qwen-rapid-aio"]})

    def test_video_uses_selected_reference_and_stays_in_project_gallery(self):
        body = {"project": "QA", "mode": "video", "refs": ["ref.png"], "video_prompt": "Turn and wave.",
                "video_size": "fast", "video_steps": 4, "seconds": 5, "seed": 0, "count": 1}
        self.assertEqual(server.queue(body)["queued"], 1)
        job = server.QUEUE[0]
        self.assertEqual(job["seed"], 0)
        self.assertIn("<Picture 1>", job["prompt"])
        self.assertEqual((job["width"], job["height"]), (768, 416))
        self.assertTrue(job["output_file"].endswith(".mp4"))
        out = Path(job["output_file"]); out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(b"test")
        job.update(status="done", stage="done")
        server.append_job(job)
        gallery = server.jobs("QA")["jobs"]
        self.assertEqual(gallery[0]["mode"], "video")
        self.assertTrue(gallery[0]["images"][0]["filename"].endswith(".mp4"))
        with self.assertRaises(HTTPException):
            server.queue({**body, "seconds": 3})

    def test_extra_reference_tokens_and_loading_progress(self):
        text = server.model_prompt("{REF2} stands behind the reference character", minimax_bridge.IMAGE_MODEL,
                                   ["ref.png", "ref2.png"])
        self.assertEqual(text, "<Picture 2> stands behind the character in <Picture 1>")
        stage, data = minimax_bridge.progress("Loading weights: 20%|##| 200/1000")
        self.assertIn("loading", stage)
        stage, data = minimax_bridge.progress("Loading checkpoint shards: 33%|###| 1/3")
        self.assertIn("loading", stage)
        stage, data = minimax_bridge.progress("Loading weights: 100%|##| 1000/1000\n[05:28:08] render clip\n")
        self.assertEqual(stage, "render clip")
        stage, data = minimax_bridge.progress("Loading weights: 100%|##| 1000/1000\n 25%|##| 1/4")
        self.assertEqual(stage, "sampling 1/4")
        stage, data = minimax_bridge.progress("|####| 11/11 - 19.07MB/s")
        self.assertIn("loading", stage)
        self.assertEqual(data["step"], 0)


if __name__ == "__main__":
    unittest.main()
