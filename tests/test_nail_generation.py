import unittest
import numpy as np
from src.nail_generation import generate_nails

class TestNailGeneration(unittest.TestCase):
    def test_generate_nails(self):
        # Create a dummy detail map
        detail_map = np.zeros((100, 100), dtype=np.float32)
        detail_map[50:60, 50:60] = 1.0 # A high detail area
        nails = generate_nails(detail_map, base_count=100, detail_multiplier=2.0)
        # Expect at least the perimeter nails + 100 inside
        # Perimeter nails: top(100) + bottom(100) + left(98) + right(98) = 396 approx
        # Plus 100 inside
        self.assertTrue(len(nails) >= 496, f"Expected at least 496 nails, got {len(nails)}")
        # Check that nails is a list of tuples (x, y)
        self.assertTrue(all(isinstance(n, tuple) and len(n) == 2 for n in nails))

if __name__ == '__main__':
    unittest.main()
