import unittest
import numpy as np
from classificatiton_utils import knn_compute_classes


class KnnTest(unittest.TestCase):
    def test_knn(self):
        nearest_indices = np.array([
            [3, 0, 2, 1, 4],
            [0, 1, 3, 4, 2],
        ])
        classes = np.array([1, 0, 2, 0, 2])

        predicted_classes = knn_compute_classes(nearest_indices, classes)
        print(predicted_classes)


if __name__ == '__main__':
    unittest.main()
