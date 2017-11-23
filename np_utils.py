import numpy as np

def expand_dims_0(arr):
    return np.expand_dims(arr, 0)
    
def take_multi(arr, indices):
    shifted_indices=indices
    width=shifted_indices.shape[1]
    for i, a_i in enumerate(shifted_indices):
        a_i[...]+=i*width
    arr_taken=np.take(arr, shifted_indices).reshape(arr.shape)
    return arr_taken
    
def iterable_to_array(items, func_):
    transformed=[func_(item) for item in items]
    transformed_arr=np.array(transformed)
    return transformed_arr
