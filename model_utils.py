def find_descriptor_model(model):
    computer_func_name = model["computer_func_name"]
    if computer_func_name in ["glcm", "histogram", "lbp", "vgg16", "pqcode"]:
        return model
    elif computer_func_name in ["normalize", "pq", "argsort", "knn", "accuracy"]:
        return find_descriptor_model(model["input_model"])
    elif computer_func_name in ["l2", "adc", "sdc", "accuracy_KNeighbors"]:
        return model["computer_func_params"]["base_model"]
    else:
        raise ValueError("cant find descriptor_model for {}".format(model["name"]))


def find_quantization_model(model):
    computer_func_name = model["computer_func_name"]
    if computer_func_name in ["pq"]:
        return model
    elif computer_func_name in ["pqcode"]:
        return model["computer_func_params"]["quantization_model"]
    elif computer_func_name in ["normalize", "argsort", "knn", "accuracy"]:
        return find_quantization_model(model["input_model"])
    elif computer_func_name in ["l2", "adc", "sdc", "accuracy_KNeighbors"]:
        return find_quantization_model(model["computer_func_params"]["base_model"])
    else:
        raise ValueError("cant find quantization_model for {}".format(model["name"]))


def find_image_path(model):
    if model["type"] == "computer":
        if "computer_func_params" in model and "image_path" in model["computer_func_params"]:
            return model["computer_func_params"]["image_path"]
        elif "input_model" in model:
            return find_image_path(model["input_model"])
    raise ValueError("cant find image_path for {}".format(model["name"]))

def find_tiles_rects_model(model):
    if model["type"] == "computer":
        if "computer_func_name" in model and  model["computer_func_name"]=="tiles_rects":
            return model
        elif "input_model" in model:
            return find_tiles_rects_model(model["input_model"])
    raise ValueError("cant find tiles_rects_model for {}".format(model["name"]))
