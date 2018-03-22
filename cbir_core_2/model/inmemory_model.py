import collections

from cbir_core_2.model.model import Model


class InmemoryModel(Model):

    def __init__(self, type, name=None, shape_dim=None, shape=None, value=None) -> None:
        super().__init__(type, name)
        self.type = type
        self.shape_dim = shape_dim
        self.shape = shape
        self.value = value


class NdarrayModel(InmemoryModel):

    def __init__(self, shape_dim, shape, value=None) -> None:
        super().__init__("ndarray", shape_dim, shape, value)


class IterableModel(InmemoryModel):

    def __init__(self, shape_dim, shape=None, value: collections.Iterable = None) -> None:
        super().__init__("iterable", shape_dim, shape)
        self.value = value


class ObjectModel(InmemoryModel):

    def __init__(self, value=None, shape_dim=0) -> None:
        super().__init__("object", None, shape_dim, None, value)
