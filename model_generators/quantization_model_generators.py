from model_generators.ds_model_generators import add_output_model


def generate_pq_model(input_model, n_quantizers, n_clusters, db_path=None, ds_name=None, dtype="float", **kwargs):
    name_ = "pq_{}_{}__{}".format(n_quantizers, n_clusters, input_model["name"])
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "pq",
        "computer_func_params": {
            "n_quantizers": n_quantizers,
            "n_clusters":n_clusters,
            "library_func_kwargs":{
                **kwargs
            }
        },
        "dtype": dtype,
        "chunk_size":-1,
        #computer assumes input as
        "input_model": input_model,
    }
    add_output_model(model, db_path, ds_name)
    return model

if __name__ == '__main__':
    pass