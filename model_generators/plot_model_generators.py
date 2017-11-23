import sys, json
import numpy as np
from ds_utils import add_output_model
import ds_utils


def generate_accuracy_plot_model(accuracy_model_list, group_type, figure_list_filter=None, title="", n_nearest=None,
                                 **kwargs):
    if group_type == "descriptor_type":
        def figure_extractor(ymodel):
            descriptor_model = ymodel["input_model"]["input_model"]["input_model"]["computer_func_params"]["base_model"]
            # shape = ds_utils.read_attrs(ymodel["output_model"])["shape"]
            y = list(ds_utils.read_array(ymodel["output_model"]))
            x = list(range(1,len(y)))
            if n_nearest:
                x = x[0:n_nearest]
                y = y[0:n_nearest]
            return {
                "label": descriptor_model["name"],
                "x": x,
                "y": y,
            }

        plot_model = generate_plot_model(accuracy_model_list, "accuracy", "n_nearest", "descriptor_type",
                                         figure_extractor, figure_list_filter, None, "", title, legend_loc="upper right",
                                         **kwargs)
        return plot_model


def generate_plot_model(y_model_list, y_label, x_label, figures_label, figure_extractor, figure_list_filter,
                        subplotvalue_extractor,
                        subplot_label, title, **kwargs):
    subplotvalue_ymodel = {}
    if subplotvalue_extractor:
        for y_model in y_model_list:
            subplotvalue = subplotvalue_extractor(y_model)
            subplots = subplotvalue_ymodel.get(subplotvalue, [])
            subplots.append(y_model)
            subplotvalue_ymodel[subplotvalue] = subplots
    else:
        subplotvalue_ymodel[""] = y_model_list

    subplot_models = []
    for subplotvalue in subplotvalue_ymodel:
        ymodels = subplotvalue_ymodel[subplotvalue]
        figures = []
        for ymodel in ymodels:
            figure = figure_extractor(ymodel)
            figures.append(figure)

        figures=figure_list_filter(figures)

        subplot_models.append({
            "subplot_label": subplot_label,
            "figures_label": figures_label,
            "y_label": y_label,
            "x_label": x_label,
            "subplot_value": subplotvalue,
            "figures": figures
        }
        )

    name_ = "plot__{}_{}_{}_{}".format(subplot_label, figures_label, x_label, y_label)
    model = {
        "type": "computer",
        "name": name_,
        "computer_func_name": "subplots",
        "computer_func_params": {
            "subplots": subplot_models,
            "title": title,
            "library_func_kwargs": {
                **kwargs
            }
        },
        "chunk_size": -1,
    }

    return model


if __name__ == '__main__':
    y_model_list = [
        {
            "output_model": "123",
            "descriptor_type": "hhh",
            "x": 3,
            "y": 5,
            "subplotvalue": 123
        },
        {
            "output_model": "222",
            "descriptor_type": "eee",
            "x": 1,
            "y": 2,
            "subplotvalue": 454
        },
        {
            "output_model": "333",
            "descriptor_type": "qqq",
            "x": 10,
            "y": 20,
            "subplotvalue": 454
        }
    ]


    def figure_extractor(ymodel):
        return {
            "label": ymodel["descriptor_type"],
            "x": ymodel["x"],
            "y": ymodel["y"],
        }


    def subplotvalue_extractor(ymodel):
        return ymodel["subplotvalue"]


    plot_model = generate_plot_model(y_model_list, "y_label", "x_label", "descriptor_type", figure_extractor,
                                     subplotvalue_extractor, "n_nearest")
    import json

    model_json = json.dumps(plot_model, indent=4)
    print(model_json)
