"""Failure-mode checks for reference labels, pairing and TIFF header handling."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from P1.audit_data import CLASSES, label_statistics, pairing_checks, tiff_header, tile_id


class AuditChecks(unittest.TestCase):
    def test_palette_separates_clutter_ignore_and_unknown(self):
        # A noncanonical pixel must not silently become the first class.
        image = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 0], [254, 0, 0]]], dtype=np.uint8))
        stats = label_statistics(image)
        self.assertEqual(stats["class_pixels"]["5"], 1)
        self.assertEqual(stats["ignored_pixels"], 1)
        self.assertEqual(stats["unknown_pixels"], 1)
        self.assertEqual(stats["class_pixels"]["0"], 0)
        self.assertEqual(stats["palette_check"], "fail")

    def test_reference_class_counts_cover_every_pixel(self):
        image = Image.fromarray(np.array([[v[1] for v in CLASSES.values()] + [(0, 0, 0)]], dtype=np.uint8))
        stats = label_statistics(image)
        self.assertEqual(sum(stats["class_pixels"].values()) + stats["ignored_pixels"], stats["total_pixels"])
        self.assertEqual(stats["palette_check"], "pass")

    def test_wrong_reference_mode_is_not_silently_converted(self):
        with self.assertRaises(ValueError):
            label_statistics(Image.new("L", (3, 3)))

    def test_pairing_rejects_size_mismatch_and_empty_supervision(self):
        image = dict(width=256, height=256, mode="RGB", dtype="uint8", decode_ok=True)
        label = dict(width=255, height=256, palette_check="pass", total_pixels=65280,
                     ignored_pixels=65280, decode_ok=True)
        problems = pairing_checks(image, label)
        self.assertIn("image_label_size_mismatch", problems)
        self.assertIn("label_contains_no_valid_pixels", problems)

    def test_header_reader_matches_known_tiff(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "small.tif"
            Image.new("RGB", (13, 7)).save(path)
            data = tiff_header(path)
        self.assertEqual((data["width"], data["height"], data["channels"]), (13, 7, 3))

    def test_tile_identifiers_do_not_use_export_timestamps(self):
        self.assertEqual(tile_id("top_mosaic_09cm_area17_noBoundary.tif"), 17)
        self.assertEqual(tile_id("ISPRS_merge_34.tif"), 34)
        self.assertIsNone(tile_id("Vaihingen-20261028.zip"))


if __name__ == "__main__":
    unittest.main()
