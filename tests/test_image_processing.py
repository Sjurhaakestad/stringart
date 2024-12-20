import unittest
import os
import numpy as np

# We assume the code to be tested is in src/image_processing/__init__.py
from src.image_processing import load_image, to_grayscale, detect_edges, compute_detail_intensity

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image_path = 'data/img1.jpeg'
        if not os.path.exists(self.image_path):
            raise FileNotFoundError("Please ensure 'img1.jpeg' is in the 'data' folder.")
        self.img = load_image(self.image_path)
        self.gray = to_grayscale(self.img)
        self.edges = detect_edges(self.gray)
        self.detail_map = compute_detail_intensity(self.edges)

    def test_load_image(self):
        self.assertIsInstance(self.img, np.ndarray)
        self.assertTrue(self.img.size > 0)

    def test_to_grayscale(self):
        self.assertEqual(len(self.gray.shape), 2)

    def test_detect_edges(self):
        self.assertEqual(self.edges.shape, self.gray.shape)

    def test_compute_detail_intensity(self):
        self.assertEqual(self.detail_map.shape, self.gray.shape)
        self.assertTrue((self.detail_map >= 0.0).all() and (self.detail_map <= 1.0).all())

if __name__ == '__main__':
    unittest.main()
