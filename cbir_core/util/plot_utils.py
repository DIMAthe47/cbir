from cbir_core.computer.computer import Computer
from file_utils import make_if_not_exists
import itertools


def plot_sublots_model_factory(computer_func_params):
    def computer_():
        return plot_sublots_model(computer_func_params)

    return Computer(computer_)


plot_type__factory = {
    "subplots": plot_sublots_model_factory
}


def plot_sublots_model(computer_func_params):
    # print(kwargs)
    import matplotlib.pyplot as plt
    plt.close()
    dpi = 300
    # fig = plt.figure(figsize=(figsize.get('width', 20), figsize.get('height', 14)))
    fig = plt.figure(figsize=(20, 14))
    # fig = plt.figure()
    fig.suptitle(computer_func_params["title"])
    import math

    subplots = computer_func_params["subplots"]
    cols = int(math.ceil((len(subplots) ** 0.5)))
    # cols = 2
    rows = len(subplots) / cols
    # print(rows, cols)

    for i, subplot in enumerate(subplots):
        sbp = plt.subplot(rows, cols, i + 1)
        subplotname = "{}_{}".format(subplot["subplot_label"], subplot["subplot_value"])
        figures = subplot["figures"]
        kwargs = computer_func_params["library_func_kwargs"]
        plot_figures(plt, figures, subplot["x_label"], subplot["y_label"], subplotname, **kwargs)
        # label__kwargs, bar, subplotprops=subplotvalue__props.get(subplotvalue, {}), **kwargs)

    plt.tight_layout()
    # save_to_file = None
    if kwargs.get("save_to_file", None) is None:
        plt.show()
    else:
        save_to_file = kwargs["save_to_file"]
        if save_to_file == True:
            save_to_file = computer_func_params["title"] + ".jpeg"
        make_if_not_exists(kwargs["save_to_file"])
        # plt.savefig(save_to_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(kwargs["save_to_file"])


def plot_figures(subplot, figures: list, xlabel, ylabel, title='', **kwargs):
    filled_markers = itertools.cycle(('o', 'v', 's', 'p', '*', '<', 'h', '>', 'H', 'D', 'd', 'P', '^', 'X'))

    for figure in figures:
        x = figure["x"]
        y = figure["y"]
        label = figure["label"]
        # print(x, y, label)
        subplot.xticks(x)
        subplot.plot(x, y, label=label)
        # subplot.plot(x, y, label, marker=next(filled_markers), **kwargs)
        # subplot.plot(np.arange(10), np.arange(10))

    lgd = subplot.legend(fontsize=11)
    subplot.title(title)
    subplot.xlabel(xlabel)
    subplot.ylabel(ylabel)
    # subplot.yticks(np.linspace(0, 1, 41))

    subplot.grid(True)
    subplot.tight_layout()
