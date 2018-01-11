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


def find_image_model(model):
    if model["type"] == "computer":
        if "computer_func_params" in model and "image_path" in model["computer_func_params"]:
            return model["computer_func_params"]["image_path"]
        elif "computer_func_params" in model and "image_model" in model["computer_func_params"]:
            return model["computer_func_params"]["image_model"]
        elif "input_model" in model:
            return find_image_model(model["input_model"])
    raise ValueError("cant find image_model for {}".format(model["name"]))


def find_rect_tiles_model(model):
    if model["type"] == "computer":
        if "computer_func_name" in model and model["computer_func_name"] == "rect_tiles":
            return model
        elif "input_model" in model:
            return find_rect_tiles_model(model["input_model"])
    raise ValueError("cant find rect_tiles model for {}".format(model["name"]))


def find_tile_rects_model(model):
    if model["type"] == "computer":
        if "computer_func_name" in model and model["computer_func_name"] == "rect_tiles":
            return model
        elif "input_model" in model:
            return find_tile_rects_model(model["input_model"])
    raise ValueError("cant find tile model for {}".format(model["name"]))


def find_openslide_tiler_model(model, parent_model=None):
    if model["type"] == "computer":
        if "computer_func_name" in model and model["computer_func_name"] == "openslide_tiler":
            return model, parent_model
        elif "input_model" in model:
            return find_openslide_tiler_model(model["input_model"], model)
    raise ValueError("cant find tile model for {}".format(model["name"]))


def find_pilimage_to_matrix_model(model):
    if model["type"] == "computer":
        if "computer_func_name" in model and model["computer_func_name"] == "pilimage_to_matrix":
            return model
        elif "input_model" in model:
            return find_pilimage_to_matrix_model(model["input_model"]), model
    raise ValueError("cant find pilimage_to_matrix model for {}".format(model["name"]))


def find_image_path(model):
    img_path = find_image_model(model)["string"]
    return img_path


def find_downsample(model):
    openslide_tiler_model, parent_model = find_openslide_tiler_model(model)
    downsample = openslide_tiler_model["computer_func_params"]["downsample"]
    return downsample
