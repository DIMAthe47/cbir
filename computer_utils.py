from datetime import datetime

from computer import Computer
from descriptor_utils import descriptor_type__computer_factory
from image_utils import jpeg_to_matrix
import ds_utils
import numpy as np
from quantization_utils import quantization_type__computer_factory
from distance_matrix_utils import type__distance_matrix_computer_factory
from nearest_indices_utils import nearest_indices_type__computer_factory
from classificatiton_utils import classification_type__computer_factory
from score_utils import score_type__computer_factory
from plot_utils import plot_type__factory


def factorify_as_computer(func):
    def factory(computer_func_params=None):
        return Computer(func)

    return factory



type__computer_factory = {
    **descriptor_type__computer_factory,
    "jpeg_to_matrix": factorify_as_computer(jpeg_to_matrix),
    **quantization_type__computer_factory,
    **type__distance_matrix_computer_factory,
    **nearest_indices_type__computer_factory,
    **classification_type__computer_factory,
    **score_type__computer_factory,
    **plot_type__factory
}


def add_factory(computer_func_name, computer_func_factory_or_func, is_factory=True):
    if is_factory:
        type__computer_factory[computer_func_name] = computer_func_factory_or_func
    else:
        type__computer_factory[computer_func_name] = factorify_as_computer(computer_func_factory_or_func)


def compute_model(model, verbose=1, force=False):
    """
        решить inmemory или ds можно было бы по наличию/отсутствию output_model: если output_model нет, то это inmemory_computer.
        но лучше чтоб был и для inmemory output_model, тогда можно будет пробовать кэшировать(например, складывать output_model от
        inmemory_computer в dict по имени output_model["name"]
    :param model:
    :return:
    """

    if model["type"] == "ds":
        outputs = ds_utils.read_array(model)
        return outputs

    if not force and "output_model" in model:
        try:
            attrs = ds_utils.read_attrs(model["output_model"])
            if "shape" in attrs:
                if verbose >= 1:
                    print("model computation skipped (force=False): {} ".format(model["name"]))
                return
        except ds_utils.DSNotFoundError:
            pass

    if verbose >= 1:
        print("model computation start: {}".format(model["name"]))

    start_datetime = datetime.now()

    computer_factory = type__computer_factory[model["computer_func_name"]]
    # print(computer_factory)
    if "computer_func_params" in model:
        computer_params = model["computer_func_params"]
        computer = computer_factory(computer_params)
    else:
        computer = computer_factory()

    # print(computer_factory)
    # print(computer)
    if "input_model" in model:
        input_model = model["input_model"]
        if "output_model" in input_model:
            inputs = ds_utils.read_array(input_model["output_model"])
        else:
            inputs = compute_model(input_model)

        n_inputs = len(inputs)
        shape = computer.get_shape()
        if shape:
            outputs = np.empty((n_inputs, *shape), model["dtype"])
        else:
            outputs = [0] * n_inputs

        if "chunk_size" in model:
            chunk_from = 0
            chunk_to = model["chunk_size"]
            if chunk_to == -1:
                chunk_to = n_inputs
            while chunk_from < n_inputs:
                if chunk_to >= n_inputs:
                    chunk_to = n_inputs
                outputs[chunk_from:chunk_to] = computer.compute(inputs[chunk_from:chunk_to])
                chunk_from = chunk_to
                chunk_to += model["chunk_size"]
        else:
            for i, source_ in enumerate(inputs):
                outputs[i] = computer.compute(source_)

        if not shape:
            outputs = np.array(outputs)

    else:
        outputs = computer.compute()

    datetime_delta = datetime.now() - start_datetime
    if verbose >= 1:
        print("model computed: {} in {} seconds".format(model["name"], datetime_delta.total_seconds()))

    if "output_model" in model:
        ds_utils.save_array(outputs, model["output_model"], attrs={"model": model})

    else:
        return outputs


#

def compute_models(model_list, verbose=1, force=False):
    for model in model_list:
        compute_model(model, verbose, force)
