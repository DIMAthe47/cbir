import numpy as np
from keras import backend as K
import image_form_utils
import itertools_utils
import operator

from cbir_core.factory.keras_utils_factories import activations_factory


def get_activations(model, X, layer_name=None, print_shape_only=True):
#returns list of shape [n_layers, len(X), layer_shape]
    print('----- activations -----')
    activations = []
    model_input = model.input

    if not isinstance(model_input, list):
        # only one input! let's wrap it in a list.
        model_input= [model_input]
        model_multi_inputs_cond = False
        list_inputs = [X, 0.]
    else:
        list_inputs = []
        list_inputs.extend(X)
        list_inputs.append(0.)

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(model_input + [K.constant(0)], [out]) for out in outputs]  # evaluation functions

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def images_stream_to_activations_arr(model, input_stream, layer_name, chunk_size=20):
#input_stream - stream of filepathes | imagebytes | PIL.Image`s
    images_arr_stream = map(image_form_utils.img_to_numpy_array, input_stream)
#    images_arr_stream = itertools.islice(images_arr_stream, 5)
#    print(list(map(np.shape, images_arr_stream)))
    images_arr_chunks_stream = itertools_utils.chunkify(images_arr_stream, chunk_size)
    cnn_activations_func = activations_factory(model, layer_name)
    cnn_activations_chunks_stream = map(np.array, map(cnn_activations_func, images_arr_chunks_stream))
    #get only one layer
    cnn_activations_chunks_stream = map(operator.itemgetter(0), cnn_activations_chunks_stream)
#     for i in cnn_activations_chunks_stream:
#         print(i.shape)
#     cnn_activations_chunks_stream=map(np.array,cnn_activations_chunks_stream))
    activations = list(cnn_activations_chunks_stream)
    activations_arr = np.concatenate(activations)
    return activations_arr
