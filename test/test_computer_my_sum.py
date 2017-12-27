import unittest

import numpy as np

from cbir_core.computer import computer_utils


def data_comes_from_some_place_other_than_output_of_another_model():
    return np.arange(10)

def my_sum(arr):
    return np.sum(arr)

class MySumTest(unittest.TestCase):
    def test_compute(self):
        computer_utils.add_factory("data_comes_from_some_place_other_than_output_of_another_model",
                                   data_comes_from_some_place_other_than_output_of_another_model, False)
        computer_utils.add_factory("my_sum", my_sum, False)

        model = {
            "type": "computer",
            "computer_func_name": "my_sum",
            "input_model": {
                "type": "inmemory_computer",
                "computer_func_name": "data_comes_from_some_place_other_than_output_of_another_model"
            }
        }

        output= computer_utils.compute_model(model)

        self.assertEqual(output, 45)

if __name__ == '__main__':
    unittest.main()

