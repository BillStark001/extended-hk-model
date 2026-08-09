import unittest

import numpy as np

from theory.mesoscopic.cli_utils import (
    first_crossing,
    first_crossing_or_nan,
    pathway_label,
)


class KineticCliUtilityTests(unittest.TestCase):
    def test_first_crossing_preserves_zero(self):
        time = np.array([0.0, 1.0, 2.0])
        values = np.array([0.5, 0.6, 0.7])
        self.assertEqual(first_crossing(time, values), 0.0)
        self.assertEqual(first_crossing_or_nan(time, values), 0.0)

    def test_missing_crossing_and_pathway_labels(self):
        time = np.array([0.0, 1.0])
        values = np.array([0.1, 0.2])
        self.assertIsNone(first_crossing(time, values))
        self.assertTrue(np.isnan(first_crossing_or_nan(time, values)))
        self.assertEqual(pathway_label(1.0, 2.0), "PbS")
        self.assertEqual(pathway_label(2.0, 1.0), "SbP")
        self.assertEqual(pathway_label(None, np.nan), "unresolved")


if __name__ == "__main__":
    unittest.main()
