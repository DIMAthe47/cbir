from cbir_core.computer.computer import Computer

from core.quantization.pq_quantizer import PQQuantizer


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

