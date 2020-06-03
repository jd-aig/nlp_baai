from utils_tf import torch_ga
import tensorflow as tf
import numpy as np


def loss_function(real, pred, length, loss_object,per_example=False):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    losses = loss_object(real, pred)
    mask = tf.cast(mask, dtype=losses.dtype)
    losses *= mask

    if per_example:
        return tf.reduce_sum(losses,1)
    else:
        return tf.reduce_sum(losses), tf.reduce_sum(length)


if __name__ == '__main__':
    real  = tf.ones([8,5])
    pred = tf.convert_to_tensor(np.random.random([8,5,1000]))
    length = tf.ones([80,50])
    l, ls = loss_function(real,pred,length)
    print(l,ls)
