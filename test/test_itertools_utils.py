import unittest
import numpy as np
from itertools_utils import chunkify
import itertools


class ItertoolsUtilsTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.length = 100

        self.params = [{"chunk_size": chunk_size} for chunk_size in [-1, ]]

    def get_stream_iterable(self):
        return range(self.length)

    def get_stream_generator(self):
        return (i for i in range(self.length))

    def get_stream_list(self):
        return list(range(self.length))

    def my_test_(self, numpy_array_chunks, true_chunks, chunk_stream):
        if numpy_array_chunks:
            for chunk1, chunk2 in itertools.zip_longest(chunk_stream, true_chunks):
                self.assertTrue(isinstance(chunk1, np.ndarray))
                self.assertTrue(np.array_equal(chunk1, chunk2))
        else:
            self.assertSequenceEqual(true_chunks, list(chunk_stream))

    def mytest_chunkify_(self, chunk_size, numpy_array_chunks, true_chunks):
        chunk_stream_iterable = chunkify(self.get_stream_iterable(), chunk_size, numpy_array_chunks)
        chunk_stream_generator = chunkify(self.get_stream_generator(), chunk_size, numpy_array_chunks)
        chunk_stream_list = chunkify(self.get_stream_list(), chunk_size, numpy_array_chunks)
        self.my_test_(numpy_array_chunks, true_chunks, chunk_stream_iterable)
        self.my_test_(numpy_array_chunks, true_chunks, chunk_stream_generator)
        self.my_test_(numpy_array_chunks, true_chunks, chunk_stream_list)

    def test_chunkify_1(self):
        true_chunks = [(i,) for i in range(100)]
        self.mytest_chunkify_(1, False, true_chunks)

    def test_chunkify_10(self):
        true_chunks = [tuple(i * 10 + j for j in range(10)) for i in range(10)]
        self.mytest_chunkify_(10, False, true_chunks)

    def test_chunkify_minus_1(self):
        true_chunks = [tuple(j for j in range(self.length))]
        self.mytest_chunkify_(-1, False, true_chunks)

    def test_chunkify_1_True(self):
        true_chunks = [np.array((i,)) for i in range(100)]
        self.mytest_chunkify_(1, True, true_chunks)

    def test_chunkify_10_True(self):
        true_chunks = [np.array(tuple(i * 10 + j for j in range(10))) for i in range(10)]
        self.mytest_chunkify_(10, True, true_chunks)

    def test_chunkify_minus_1_True(self):
        true_chunks = [np.array(tuple(j for j in range(self.length)))]
        self.mytest_chunkify_(-1, True, true_chunks)


if __name__ == '__main__':
    unittest.main()
