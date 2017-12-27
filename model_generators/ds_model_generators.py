def generate_ds_model(db_path=None, ds_name=None):
    model = {
        "type": "ds",
        "db_path": db_path,
        "ds_name": ds_name
    }
    return model


def add_output_model(input_model, db_path=None, ds_name=None):
    if db_path:
        if not ds_name:
            ds_name = "{}/{}".format(input_model["computer_func_name"], input_model["name"])
        output_model = generate_ds_model(db_path, ds_name)
        input_model["output_model"] = output_model
    else:
        input_model["output_model"] = {
            "type": "inmemory"
        }
    return input_model
