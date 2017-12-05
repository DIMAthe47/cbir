import itertools
from functools import partial

import numpy as np


def chunkify(it, chunk_size, numpy_array_chunks=True):
    if chunk_size == -1:
        it_list = list(it)
        if len(it_list) == 1 and len(it_list[0].shape) == 1:
            chunks_stream = [it_list]
        else:
            chunks_stream = it_list
    else:
        tw = itertools.takewhile
        count = itertools.count
        islice = itertools.islice
        it_iter = iter(it)
        # print(next(it_iter).shape)
        # for i in it_iter:
        #     print(i.shape)
        chunks_stream = tw(bool, (tuple(islice(it_iter, chunk_size)) for _ in count()))

    if numpy_array_chunks:
        chunks_stream = map(partial(np.array, copy=False), chunks_stream)
    return chunks_stream


def main():
    it = range(100)
    chunk_stream = chunkify(it, -1)
    for chunk in chunk_stream:
        print(chunk)


if __name__ == '__main__':
    main()
