class Computer():
    def __init__(self, computer_func, shape=None):
        self.shape = shape
        self.computer_func = computer_func

    def get_shape(self):
        return self.shape

    def compute(self, *args):
        return self.computer_func(*args)
