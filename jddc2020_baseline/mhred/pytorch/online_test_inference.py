from solver import Solver
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import os
import pickle
import re


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='test')

    print('Loading Vocabulary...')
    vocab = Vocab(lang="zh")
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size
    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        images=load_pickle(config.images_path),
        conv_img_length=load_pickle(config.images_len_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=1,
        shuffle=False)

    solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)

    solver.build()
    solver.generate_for_evaluation()