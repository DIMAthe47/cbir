from cbir_core.computer.computer import Computer


def factorify_as_computer(func):
    def factory(computer_func_params=None):
        return Computer(func)

    return factory