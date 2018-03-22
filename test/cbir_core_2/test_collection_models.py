import unittest

from cbir_core.util.ds_utils import to_json
from cbir_core_2.model.computer_model import ObjectCollectionToObjectModelCollection
from cbir_core_2.model.inmemory_model import IterableModel


class MySumTest(unittest.TestCase):
    def test_compute(self):
        ids_model: IterableModel = IterableModel(0, None, list(range(10)))
        idmodels_model = ObjectCollectionToObjectModelCollection(ids_model)
        idmodels_model_computed = idmodels_model.compute()
        print(to_json(ids_model))
        print(to_json(idmodels_model_computed))


if __name__ == '__main__':
    unittest.main()
