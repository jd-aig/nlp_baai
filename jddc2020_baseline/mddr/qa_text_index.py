import jieba
from gensim.models import KeyedVectors
import numpy as np
import argparse
from indexers.faiss_indexers import *
from tqdm import tqdm
from utils.doc import *


def create_text_embedding_idx(w2v_file, qa_file, idx_file):
 
    index = DenseFlatIndexer(200)

    model = KeyedVectors.load_word2vec_format(w2v_file)

    with open(qa_file) as f:
        lines = f.readlines()

    buffer = []
    for id, line in enumerate(tqdm(lines)):
        word = line.strip().split('\t')
        ques = word[0]

        ques_vec = get_sentence_emb(model, ques)
        buffer.append((id, ques_vec))

        if 0 < 50000 == len(buffer):
            index.index_data(buffer)
            buffer = []

    index.index_data(buffer)

    index.serialize(idx_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to create embedding index")

    parser.add_argument('-e', '--w2v_file', default='1000000-small.txt')
    parser.add_argument('-f', '--qa_file', default='QA_dbs.txt')
    parser.add_argument('-i', '--index_file', default='jddc')

    args = parser.parse_args()

    create_text_embedding_idx(args.w2v_file, args.qa_file, args.index_file)
