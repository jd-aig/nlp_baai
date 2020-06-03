import tensorflow as tf


def pad(tensor, length):

    if length > tensor.shape[0]:
        ii = list(tensor.shape)
        ii[0] = length
        p = tf.zeros(ii)[:(length - tensor.shape[0])]

        return tf.concat((tensor, p),0)
    else:
        return tensor


def pad_and_pack(tensor_list):
    length_list = ([t.shape[0] for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = tf.stack(padded, 0)
    return packed, length_list
