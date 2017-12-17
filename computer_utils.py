import types
from datetime import datetime

import numpy as np
import collections

import ds_utils
from classificatiton_utils import classification_type__computer_factory
from descriptor_utils import descriptor_type__computer_factory
from distance_matrix_utils import type__distance_matrix_computer_factory
from factory_utils import factorify_as_computer
from image_utils import image_transform_type__computer_factory
from itertools_utils import chunkify
from nearest_indices_utils import nearest_indices_type__computer_factory
from np_utils import iterable_to_array
from plot_utils import plot_type__factory
from quantization_utils import quantization_type__computer_factory
from score_utils import score_type__computer_factory
from slide_utils import image_util_type__computer_factory

type__computer_factory = {
    **image_transform_type__computer_factory,
    **descriptor_type__computer_factory,
    **quantization_type__computer_factory,
    **type__distance_matrix_computer_factory,
    **nearest_indices_type__computer_factory,
    **classification_type__computer_factory,
    **score_type__computer_factory,
    **plot_type__factory,
    **image_util_type__computer_factory
}


def add_factory(computer_func_name, computer_func_factory_or_func, is_factory=True):
    if is_factory:
        type__computer_factory[computer_func_name] = computer_func_factory_or_func
    else:
        type__computer_factory[computer_func_name] = factorify_as_computer(computer_func_factory_or_func)


def read_input_model(input_model, verbose=1):
    if verbose >= 3:
        print("read_input_model", input_model)
    inputs = None
    if input_model["type"] == "ds":
        inputs = ds_utils.read_array(input_model)
    elif input_model["type"] == "string":
        inputs = input_model["string"]
    # elif input_model["type"] == "slide_image":
    #     inputs = OpenSlide(input_model["image_path"])
    elif input_model["type"] == "list":
        inputs = input_model["list"]
    elif input_model["type"] == "inmemory":
        inputs = compute_model(input_model, force=True, verbose=0)
    elif input_model["type"] == "computer":
        if input_model["output_model"]["type"] == "inmemory":
            inputs = compute_model(input_model, force=True, verbose=0)
        else:
            inputs = compute_model(input_model, force=True, verbose=0)
    return inputs


def stop_recompute_if_not_force(model, force=False, verbose=1):
    if not force and "output_model" in model and model["output_model"]["type"] == "ds":
        try:
            attrs = ds_utils.read_attrs(model["output_model"])
            if "shape" in attrs:
                if verbose >= 1:
                    print("model computation skipped (force=False): {} ".format(model["name"]))
                return True
        except (ds_utils.DSNotFoundError, OSError):
            pass
    return False


# def compute_for_input_as_ndarray(model):

def build_computer(model):
    computer_factory = type__computer_factory[model["computer_func_name"]]
    # print(computer_factory)
    if "computer_func_params" in model:
        computer_params = model["computer_func_params"]
        computer = computer_factory(computer_params)
    else:
        computer = computer_factory()

    # print(computer)
    return computer


def compute_outputs(model, verbose=1):
    computer = build_computer(model)

    if "input_model" in model:
        input_ = read_input_model(model["input_model"], verbose)

        if isinstance(input_, np.ndarray):
            # if hasattr(input_, "__len__"):
            n_inputs = len(input_)
            shape = computer.get_shape()
            if shape:
                outputs = np.empty((n_inputs, *shape), model["dtype"])
            else:
                outputs = [0] * n_inputs
            # else:
            #     outputs = []

            if "chunk_size" in model:
                chunk_from = 0
                chunk_to = model["chunk_size"]
                if chunk_to == -1:
                    chunk_to = n_inputs
                while chunk_from < n_inputs:
                    if chunk_to >= n_inputs:
                        chunk_to = n_inputs
                    outputs[chunk_from:chunk_to] = computer.compute(input_[chunk_from:chunk_to])
                    chunk_from = chunk_to
                    chunk_to += model["chunk_size"]
            else:
                for i, source_ in enumerate(input_):
                    outputs[i] = computer.compute(source_)
                    # if not shape:
                    #     outputs = np.array(outputs)
        elif isinstance(input_, str):
            outputs = computer.compute(input_)
        elif isinstance(input_, collections.Iterable):
            if "chunk_size" in model:
                outputs = []
                input_chunks_iter = chunkify(input_, chunk_size=model["chunk_size"])
                # print(next(input_chunks_iter))
                for inputs_chunk in input_chunks_iter:
                    output_chunks = computer.compute(inputs_chunk)
                    outputs.append(output_chunks)
            else:
                outputs = map(computer.compute, input_)

        elif input_:
            outputs = computer.compute(input_)
    else:
        outputs = computer.compute()
    return outputs


def save_outputs(outputs, model):
    output_model = model["output_model"]
    if output_model["type"] == "ds":
        if isinstance(outputs, np.ndarray):
            pass
        elif isinstance(outputs, collections.Iterable):
            if "chunk_size" in model:
                print(outputs)
                # print(next(outputs))
                outputs = np.concatenate(outputs)
            elif isinstance(outputs, list):
                outputs = np.array(outputs, copy=False)
            else:
                outputs = [output_ for output_ in outputs]
                outputs = np.array(outputs, copy=False)
        else:
            raise ValueError("Can`t save outputs of type {} to ds".format(type(outputs)))
        ds_utils.save_array(outputs, output_model, attrs={"model": model})
    elif output_model["type"] == "inmemory":
        pass
    return outputs


def compute_model(model, force=False, verbose=1):
    """
        решить inmemory или ds можно было бы по наличию/отсутствию output_model: если output_model нет, то это inmemory_computer.
        но лучше чтоб был и для inmemory output_model, тогда можно будет пробовать кэшировать(например, складывать output_model от
        inmemory_computer в dict по имени output_model["name"]
    :param model:
    :return:
    """
    if verbose >= 3:
        print("compute_model", model)
    stop_ = stop_recompute_if_not_force(model, force, verbose)
    if stop_:
        outputs = read_input_model(model["output_model"])
        return outputs

    if verbose >= 1:
        if "name" in model:
            print("model computation start: {}".format(model["name"]))

    start_datetime = datetime.now()

    outputs = compute_outputs(model)

    outputs = save_outputs(outputs, model)

    datetime_delta = datetime.now() - start_datetime
    if verbose >= 1:
        print("model computed: {} in {} seconds".format(model["name"], datetime_delta.total_seconds()))

    return outputs


def compute_models(model_list, force=False, verbose=1):
    for model in model_list:
        compute_model(model, force, verbose)
