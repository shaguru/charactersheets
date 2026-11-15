import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mvm import engine_image, forge_bridge


class ImageBridgeTests(unittest.TestCase):
    def test_generated_image_is_renamed_into_forge_gallery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "gallery" / "job.png"
            reference = root / "reference.png"
            reference.write_bytes(b"reference")
            request = root / "request.json"
            request.write_text(json.dumps({"kind": "image", "id": "job", "prompt": "A character sheet.",
                "refs": [str(reference)], "output": str(output), "width": 512, "height": 512, "seed": 42}))

            def generate(prompt, refs, folder, size, **kwargs):
                self.assertEqual(refs, [reference])
                self.assertEqual(size, (512, 512))
                self.assertEqual(kwargs["seed"], 42)
                generated = folder / "job_42.png"
                generated.write_bytes(b"rendered image")
                return [generated]

            with patch.object(engine_image, "generate", side_effect=generate), \
                    patch.object(forge_bridge.gpulock, "gpu_job", return_value=contextlib.nullcontext()):
                forge_bridge.run(str(request))
            self.assertEqual(output.read_bytes(), b"rendered image")
            self.assertFalse((output.parent / "job_42.png").exists())


if __name__ == "__main__":
    unittest.main()
