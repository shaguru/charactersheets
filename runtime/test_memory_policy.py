import copy
import json
import unittest
from pathlib import Path

from memory_policy import LOW_VRAM_PIXELS, limit_dimensions, presets_for_profile, profile_for_vram


class MemoryPolicyTests(unittest.TestCase):
    def test_3070_uses_low_vram_but_3090_keeps_standard_sizes(self):
        self.assertTrue(profile_for_vram(8 * 1024**3)["low_vram"])
        self.assertTrue(profile_for_vram(12 * 1024**3)["low_vram"])
        self.assertFalse(profile_for_vram(24 * 1024**3)["low_vram"])
        self.assertFalse(profile_for_vram(0)["low_vram"])

    def test_explicit_override_and_invalid_setting(self):
        self.assertTrue(profile_for_vram(24 * 1024**3, "1")["low_vram"])
        self.assertFalse(profile_for_vram(8 * 1024**3, "0")["low_vram"])
        with self.assertRaises(ValueError):
            profile_for_vram(8 * 1024**3, "maybe")

    def test_presets_fit_low_vram_without_mutating_catalog(self):
        presets = json.loads(Path(__file__).with_name("presets.json").read_text(encoding="utf-8"))
        original = copy.deepcopy(presets)
        adjusted = presets_for_profile(presets, profile_for_vram(8 * 1024**3))
        self.assertEqual(presets, original)
        self.assertEqual(adjusted["sizes"]["match"], presets["sizes"]["match"])
        for size in adjusted["sizes"].values():
            self.assertLessEqual(size["w"] * size["h"], LOW_VRAM_PIXELS)

    def test_dimensions_include_extreme_aspect_ratios_and_match_reference(self):
        for width, height in [(1792, 768), (1024, 1024), (2048, 256), (256, 2048)]:
            w, h = limit_dimensions(width, height, LOW_VRAM_PIXELS)
            self.assertLessEqual(w * h, LOW_VRAM_PIXELS)
            self.assertGreaterEqual(min(w, h), 256)
            self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertEqual(limit_dimensions(512, 512, LOW_VRAM_PIXELS), (512, 512))
        self.assertEqual(limit_dimensions(1792, 768, 2_097_152), (1792, 768))


if __name__ == "__main__":
    unittest.main()
