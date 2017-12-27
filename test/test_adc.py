import unittest
import numpy as np
from cbir_core.util.distance_matrix_utils import asymmetric_distance


class ADCTest(unittest.TestCase):
    def test_adc(self):
        nearest_indices = np.array([
            [3, 0, 2, 1, 4],
            [0, 1, 3, 4, 2],
        ])
        query = np.array([1, 0, 1, 2])
        cluster_centers = np.array([
            [[0, 1], [2, 5]],
            [[1, 1], [0, 2]],
        ])
        pqcodeds = np.array([
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        )

        distances = asymmetric_distance(cluster_centers, query, pqcodeds)
        print(distances)


if __name__ == '__main__':
    unittest.main()
