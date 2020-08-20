from solver import Solver
from data_tf import Dataloader
from configs_tf import get_config
from utils_tf import Vocab
import tensorflow as tf
import os
import pickle
import re


def loadpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu0 = gpus[1]  # 如果有多个GPU，仅使用第0个GPU
        tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
        tf.config.set_visible_devices([gpu0], "GPU")
        print('use gpu ')
    else:
        print('use cpu')
    config = get_config(mode='test')

    print('Loading Vocabulary...')
    vocab = Vocab(lang="zh")
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size
    print('loading file')
    sentences = loadpickle(config.sentences_path)
    images = loadpickle(config.images_path)
    images_len = loadpickle(config.images_len_path)
    conv_len = loadpickle(config.conversation_length_path)
    sent_len = loadpickle(config.sentence_length_path)
    print('loading done')
    data_loader = Dataloader(
        sent=sentences,
        img=images,
        img_len=images_len,
        conv_len=conv_len,
        sent_len=sent_len,
        vocab=vocab,
        batch_size=1,
        mode='test'

    )
    test_data = data_loader.get_data_loader()

    solver = Solver(config, None, test_data, vocab=vocab, is_train=False)

    solver.build()
    solver.generate_for_evaluation()

