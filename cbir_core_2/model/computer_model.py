from cbir_core_2.model.inmemory_model import IterableModel, ObjectModel
from cbir_core_2.model.model import Model


class ComputerModel(Model):

    def __init__(self, input_model=None, name=None) -> None:
        super().__init__("computer", name)
        self.input_model = input_model

    def compute(self, input_model: Model):
        pass


class ComputerFuncModel(Model):

    def __init__(self, computer_func_name, computer_func_params=None, dtype=None, input_model=None, name=None) -> None:
        super().__init__("computer", name)
        self.computer_func_name = computer_func_name
        self.computer_func_params = computer_func_params
        self.dtype = dtype
        self.input_model = input_model


class HistogramComputerModel(ComputerFuncModel):
    def __init__(self, n_bins, density, dtype="float", input_model=None, name=None) -> None:
        computer_func_params = {
            "n_bins": n_bins,
            "density": density,
            "library_func_kwargs": {}
        }
        super().__init__("histogram", computer_func_params, dtype, input_model, name)


class ObjectCollectionToObjectModelCollection(ComputerModel):

    def __init__(self, input_model: IterableModel) -> None:
        super().__init__(input_model)

    def compute(self) -> Model:
        models = [ObjectModel(item) for item in self.input_model.value]
        new_model = IterableModel(0, value=models)
        return new_model
