"""Regression checks for the Windows FP8 checkpoint reader and reference markers."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rapid_aio import _SafeTensorReader


class ReaderTests(unittest.TestCase):
    def test_fp8_bf16_and_empty_survive_reader_close(self):
        expected = {
            "fp8": torch.tensor([[-1.5, 0, 0.25], [1, 2, 4]], dtype=torch.float8_e4m3fn),
            "bf16": torch.tensor([1.5, -3], dtype=torch.bfloat16),
            "marker": torch.empty(0),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            save_file(expected, path)
            with _SafeTensorReader(path) as reader:
                actual = {key: reader.get_tensor(key) for key in reader.keys()}
            for key, value in expected.items():
                self.assertEqual(actual[key].dtype, value.dtype)
                self.assertEqual(actual[key].shape, value.shape)
                torch.testing.assert_close(actual[key].float(), value.float())
            # Release mapped storage before deleting the temporary file on Windows.
            actual.clear()
            del reader

    def test_invalid_offsets_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            save_file({"x": torch.ones(1)}, path)
            with _SafeTensorReader(path) as reader:
                reader.info["x"].end = reader.data_size + 1
                with self.assertRaisesRegex(ValueError, "Invalid tensor offsets"):
                    reader.get_tensor("x")
            del reader


if __name__ == "__main__":
    unittest.main()
