from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import generate
import cv2
import numpy as np


class DatasetTests(unittest.TestCase):
    def test_full_coverage_overlaps_and_exact_uint16_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ground_truth.png"
            image = np.random.default_rng(7).integers(0, 65536, (313, 477), dtype=np.uint16)
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            data.tofile(source)
            original = source.read_bytes()
            metadata = generate(source)
            self.assertEqual(len(metadata["tiles"]), 16)
            self.assertEqual(len(list(root.glob("[0-9][0-9].png"))), 16)
            coverage = np.zeros(image.shape, np.uint8)
            for tile in metadata["tiles"]:
                x, y, w, h = [tile[key] for key in ("x", "y", "width", "height")]
                crop = cv2.imdecode(np.fromfile(root / tile["file"], np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(crop, image[y:y+h, x:x+w])
                coverage[y:y+h, x:x+w] += 1
            self.assertTrue(np.all(coverage > 0))
            self.assertTrue(np.any(coverage > 1))
            self.assertTrue(all(value > 0 for value in metadata["horizontal_overlap_pixels"]))
            self.assertTrue(all(value > 0 for value in metadata["vertical_overlap_pixels"]))
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "already exists"):
                generate(source)
            generate(source, overwrite=True)

    def test_invalid_overlap_is_rejected_before_reading(self):
        for overlap in (0, 1, -1, float("nan")):
            with self.assertRaises(ValueError):
                generate("missing.png", overlap=overlap)


if __name__ == "__main__":
    unittest.main()
