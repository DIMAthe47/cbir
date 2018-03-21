class InmemoryModel:

    def __init__(self, type, shape_dim, shape, value=None) -> None:
        super().__init__()
        self.type = type
        self.shape_dim = shape_dim
        self.shape = shape
        self.value = value


class NdarrayModel(InmemoryModel):

    def __init__(self, shape_dim, shape, value=None) -> None:
        super().__init__("ndarray", shape_dim, shape, value)


class IterableModel(InmemoryModel):

    def __init__(self, shape_dim, shape=None, value=None) -> None:
        super().__init__("iterable", shape_dim, shape, value)


class ObjectModel(InmemoryModel):

    def __init__(self, shape_dim=0, shape=None, value=None) -> None:
        super().__init__("object", shape_dim, shape, value)
