"""Verify coverage and separation of the complete labeled dataset."""
import copy
import unittest

from P1.audit_data import EXPECTED_LABELED_IDS
from P1.freeze_full_split import PARTITIONS, canonical_full_label, validate_full_split


class FullSplitChecks(unittest.TestCase):
    def setUp(self):
        self.records = {n: {"pair_status": "pass"} for n in EXPECTED_LABELED_IDS}

    def test_full_labeled_set_covered_once(self):
        validate_full_split(PARTITIONS, self.records)
        self.assertEqual([len(PARTITIONS[k]) for k in ("train", "validation", "test")], [10, 2, 4])

    def test_overlapping_test_tile_rejected(self):
        bad = copy.deepcopy(PARTITIONS)
        bad["train"][0] = 30
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_full_split(bad, self.records)

    def test_unlabeled_tile_cannot_replace_reference_tile(self):
        bad = copy.deepcopy(PARTITIONS)
        bad["train"][0] = 24
        with self.assertRaisesRegex(ValueError, "all 16"):
            validate_full_split(bad, self.records)

    def test_failed_reference_is_rejected(self):
        self.records[23]["pair_status"] = "review_required"
        with self.assertRaisesRegex(ValueError, "passing"):
            validate_full_split(PARTITIONS, self.records)

    def test_identical_copy_name_is_not_canonical(self):
        record = {"full_label_paths": ["part2/top_mosaic_09cm_area23(1).tif", "part1/top_mosaic_09cm_area23.tif"]}
        self.assertEqual(canonical_full_label(record), "part1/top_mosaic_09cm_area23.tif")


if __name__ == "__main__":
    unittest.main()
