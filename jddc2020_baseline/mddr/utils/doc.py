import jieba
import numpy as np


def tokenize_spt(text):

    sp_token = ['<img>', '<url>', '<sos>', '<eos>', '<num>']

    resp_list = list()
    tmp_list = jieba.cut(text, cut_all=False)

    seg_list = list(tmp_list)
    i = 0

    while i < len(seg_list):
        if ''.join(seg_list[i:i + 3]) in sp_token:
            resp_list.append(''.join(seg_list[i:i + 3]))
            i = i + 3
        else:
            resp_list.append(''.join(seg_list[i]))
            i = i + 1

    return resp_list


def get_sentence_emb(model, text):

    token_list = tokenize_spt(text)
    z = np.zeros(200)
    count = 0

    for token in token_list:
        if model.__contains__(token):
            a = model.get_vector(token)
            z = a + z
            count = count + 1

    if count == 0:
        count = 1

    return (z/count).astype('float32')
