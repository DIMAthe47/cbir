from cbir_core_2.model.model import Model


class ComputerModel(Model):

    def __init__(self, computer_func_name, computer_func_params=None, dtype=None, input_model=None, name=None) -> None:
        super().__init__("computer", name)
        self.computer_func_name = computer_func_name
        self.computer_func_params = computer_func_params
        self.dtype = dtype
        self.input_model = input_model
