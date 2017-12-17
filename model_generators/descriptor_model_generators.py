import sys, json
import numpy as np
from ds_utils import add_output_model


def generate_histogram_model(input_model, n_bins, density=True, db_path=None, ds_name=None, dtype="float"):
    name_ = "histogram_{}".format(n_bins)
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "histogram",
        "computer_func_params": {
            "n_bins": n_bins,
            "density": density,
            "library_func_kwargs": {}
        },
        "dtype": dtype,
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_lbp_model(input_model, P, R, method, db_path=None, ds_name=None, dtype="float"):
    name_ = "lbp_{}_{}_{}".format(P, R, method)
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "lbp",
        "computer_func_params": {
            "library_func_kwargs": {
                "P": P,
                "R": R,
                "method": method
            }
        },
        "dtype": dtype,
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_vgg16_model(input_model, layer_name, chunk_size=30, db_path=None, ds_name=None, dtype="float"):
    name_ = "vgg16_{}".format(layer_name)
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "vgg16",
        "computer_func_params": {
            "layer_name": layer_name,
            "library_func_kwargs": {}
        },
        "dtype": dtype,
        "input_model": input_model,
        "chunk_size": chunk_size
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_pqcode_model(input_model, quantization_model, db_path=None, ds_name=None, dtype="int"):
    name_ = "pqcode__{}".format(quantization_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "pqcode",
        "computer_func_params": {
            "quantization_model": quantization_model,
            "library_func_kwargs": {}
        },
        "dtype": dtype,
        "input_model": input_model,
        "chunk_size": 30
    }
    add_output_model(model, db_path, ds_name)
    return model


def generate_normalization_model(input_model, norm="l2", db_path=None, ds_name=None, dtype="float"):
    name_ = "{}__{}".format(norm, input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "normalize",
        "computer_func_params": {
            "norm": norm,
            "library_func_kwargs": {
            }
        },
        "dtype": dtype,
        "input_model": input_model,
        "chunk_size": -1
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    model = generate_histogram_model("in", 128, "out")
    print(model)
    model = generate_vgg16_model("in", "fc1", "out")
    print(model)
