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
