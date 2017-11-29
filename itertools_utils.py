import itertools
from functools import partial

import numpy as np


def chunkify(it, chunk_size, numpy_array_chunks=True):
    tw = itertools.takewhile
    count = itertools.count
    islice = itertools.islice
    it_iter = iter(it)
    chunks_stream = tw(bool, (tuple(islice(it_iter, chunk_size)) for _ in count()))
    if numpy_array_chunks:
        chunks_stream = map(partial(np.array, copy=False), chunks_stream)
    return chunks_stream
