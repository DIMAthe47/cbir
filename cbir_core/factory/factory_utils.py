from cbir_core.computer.computer import Computer


def factorify_as_computer(func):
    def factory(computer_func_params=None):
        return Computer(func)

    return factory

def factorify_as_computer_kw(func):
    def factory(computer_func_params=None):
        def computer_(X):
            return func(X, **computer_func_params)

        return Computer(computer_)

    return factory
