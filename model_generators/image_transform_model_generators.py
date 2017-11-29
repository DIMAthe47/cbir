import sys, json
import numpy as np
from ds_utils import add_output_model


def generate_jpeg_to_matrix_model(input_model, db_path=None, ds_name=None, dtype="float"):
    name_ = "jpeg_to_matrix"
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "jpeg_to_matrix",
        "dtype": dtype,
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model

def generate_pilimage_to_matrix_model(input_model, db_path=None, ds_name=None, dtype="float"):
    name_ = "pilimage_to_matrix"
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "pilimage_to_matrix",
        "dtype": dtype,
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
