import numpy as np
import sys
from computer import Computer

sys.path.append('/home/dimathe47/PycharmProjects/pq')
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
import ds_utils
from datetime import datetime
from copy import deepcopy


def pq_cluster_centers_computer_factory(computer_func_params):
    def computer_(descriptors):
        n_quantizers = computer_func_params["n_quantizers"]
        n_clusters = computer_func_params["n_clusters"]
        kwargs = computer_func_params["library_func_kwargs"]
        pq = PQQuantizer(n_quantizers=n_quantizers, n_clusters=n_clusters, **kwargs)
        pq.fit(descriptors)
        return pq.get_cluster_centers()

    return Computer(computer_)


quantization_type__computer_factory = {
    "pq": pq_cluster_centers_computer_factory,
}


def compute_cluster_centers(quantization_model, verbose=1):
    if verbose < 2:
        quantization_model = deepcopy(quantization_model)
        quantization_model["kwargs"]["verbose"] = False

    start_datetime = datetime.now()
    descriptors = ds_utils.read_array(quantization_model["input_ds"])
    descriptors = descriptors.reshape((len(descriptors), -1))
    computer_factory = quantization_type__computer_factory[quantization_model["type"]]
    computer_ = computer_factory(quantization_model)
    cluster_centers = computer_.compute(descriptors)
    ds_utils.save_array(cluster_centers, quantization_model["output_ds"], attrs={"model": quantization_model})
    datetime_delta = datetime.now() - start_datetime

    if verbose >= 1:
        print("model computed: {} in {} seconds".format(quantization_model["name"], datetime_delta.total_seconds()))
