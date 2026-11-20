import tempfile
import unittest
from pathlib import Path

from P1.artifact_io import write_immutable


class ImmutableArtifactChecks(unittest.TestCase):
    def test_identical_rerun_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            write_immutable(path, '{"train": [1]}\n')
            write_immutable(path, '{"train": [1]}\n')
            self.assertEqual(path.read_text(), '{"train": [1]}\n')

    def test_changed_split_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            write_immutable(path, 'original')
            with self.assertRaises(FileExistsError):
                write_immutable(path, 'different')
            self.assertEqual(path.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
