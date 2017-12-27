from model_generators.ds_model_generators import add_output_model


def generate_distance_matrix_model(input_model, base_model, metric="l2", db_path=None, ds_name=None, dtype="float",
                                   **kwargs):
    name_ = "distance_matrix_{}__{}".format(metric, base_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": metric,
        "computer_func_params": {
            "base_model": base_model,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size": -1,
        # computer assumes input as
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model


if __name__ == '__main__':
    pass
