"""Offline component loading must not resolve sharded VAEs through Hub metadata."""
import unittest
from unittest.mock import Mock, patch

from mvm import models, render


class OfflineVideoTests(unittest.TestCase):
    def renderer(self):
        worker = render.Renderer.__new__(render.Renderer)
        worker.repo = "C:/installed/snapshots/pinned-revision"
        worker.s = {}
        worker.say = Mock()
        return worker

    def test_all_video_components_load_from_installed_snapshot(self):
        worker = self.renderer()
        pipe = Mock()
        workflow = Mock(sub_blocks={"text_encoder": object()})
        workflow.init_pipeline.return_value = pipe
        with patch("diffusers.ModularPipeline.from_pretrained", return_value=Mock(blocks=workflow)), \
                patch.object(render.offload, "apply"):
            self.assertIs(worker.load_video(), pipe)
        self.assertEqual(pipe.load_components.call_count, 3)
        for call in pipe.load_components.call_args_list:
            self.assertEqual(call.kwargs["pretrained_model_name_or_path"], worker.repo)
            self.assertTrue(call.kwargs["local_files_only"])

    def test_component_failure_has_actionable_error(self):
        worker = self.renderer()
        pipe = Mock(vae=None)
        workflow = Mock(sub_blocks={"text_encoder": object()})
        workflow.init_pipeline.return_value = pipe
        with patch("diffusers.ModularPipeline.from_pretrained", return_value=Mock(blocks=workflow)), \
                self.assertRaisesRegex(RuntimeError, "installed video components: vae"):
            worker.load_video()

    def test_video_snapshot_is_pinned_and_offline(self):
        repo = "abhishekchohan/minimax-h3-int8"
        with patch("huggingface_hub.snapshot_download", return_value="C:/installed") as download:
            models.snapshot(repo)
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertEqual(download.call_args.kwargs["revision"], models.VIDEO_REVISIONS[repo])


if __name__ == "__main__":
    unittest.main()
