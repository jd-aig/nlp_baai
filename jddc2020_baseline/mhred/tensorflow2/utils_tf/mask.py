import tensorflow as tf

def torch_ga(input, indice):
    len_1d, len_2d = input.shape
    idx_matrix = tf.tile(tf.expand_dims(tf.range(0, len_2d), 0), [len_1d,1])
    idx_ls = []
    for i in range(len_1d):
        idx_mask_new = tf.equal(idx_matrix, indice[i])
        idx_mask_new = tf.gather(idx_mask_new, i)
        #idx_mask_new = tf.expand_dims(idx_mask_new, 0)
#         if i:
#             idx_mask = tf.concat([idx_mask,idx_mask_new], axis=0)
#         else:
#             idx_mask = idx_mask_new
        idx_ls.append(idx_mask_new)
        print(i)
    idx_mask = tf.stack(idx_ls)
    input *= tf.cast(idx_mask, dtype=tf.float32)
    input = tf.reduce_sum(input, 1)
    return input