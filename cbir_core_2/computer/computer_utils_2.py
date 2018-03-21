from cbir_core.computer.computer import Computer
from cbir_core.factory.classification_utils_factories import classification_type__computer_factory
from cbir_core.factory.descriptor_utils_factories import descriptor_util__computer_factory
from cbir_core.factory.distance_matrix_utils_factories import type__distance_matrix_computer_factory
from cbir_core.factory.factory_utils import factorify_as_computer
from cbir_core.factory.image_utils_factories import image_transform_type__computer_factory
from cbir_core.factory.nearest_indices_utils_factories import nearest_indices_type__computer_factory
from cbir_core.factory.openslide_utils_factories import openslide_util__computer_factory
from cbir_core.factory.quantization_utils_factories import quantization_type__computer_factory
from cbir_core.factory.tiling_utils_factories import tiling_util__computer_factory
from cbir_core.util.plot_utils import plot_type__factory
from cbir_core.util.score_utils import score_type__computer_factory
from cbir_core_2.model.computer_model import ComputerModel
from cbir_core_2.model.inmemory_model import InmemoryModel

type__computer_factory = {
    **image_transform_type__computer_factory,
    **descriptor_util__computer_factory,
    **quantization_type__computer_factory,
    **type__distance_matrix_computer_factory,
    **nearest_indices_type__computer_factory,
    **classification_type__computer_factory,
    **score_type__computer_factory,
    **plot_type__factory,
    **tiling_util__computer_factory,
    **openslide_util__computer_factory
}


def build_computer2(model: ComputerModel) -> Computer:
    computer_factory = type__computer_factory[model.computer_func_name]
    # print(computer_factory)
    if model.computer_func_params is not None:
        computer = computer_factory(model.computer_func_params)
    else:
        computer = computer_factory()
    # print(computer)
    return computer


def compute_models(model_list, force=False, verbose=1):
    for model in model_list:
        compute_model(model, force, verbose)


def compute_model(model):
    if model is None:
        return None
    elif isinstance(model, InmemoryModel):
        return model
    elif isinstance(model, ComputerModel):
        computer_model: ComputerModel = model
        computer = build_computer2(computer_model)
        if computer_model.input_model is not None:
            input_model_computed = compute_model(computer_model.input_model)
            # check dims and shapes then pass to computer
        # else:
