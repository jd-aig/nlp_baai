# -*- coding: utf-8 -*-

import json
import argparse
from raw_data_preprocess import *
from pathlib import Path
from prepare_data import load_conversations, pad_sentences, images_str_2_list
import pickle
from utils import Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


# set default path for data and test data
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
images_test_dir = project_dir.joinpath('./online_test_data/images_test/')


def gen_test_set(dir, session_turn):

    f_out = open('data/'+'test.txt', 'w')

    sess_length = session_turn * 2
    with open(dir+'test_questions.json') as f:
        items = json.load(f)

    for item in items:

        img_list = list()
        src_str = ''

        ques_id = item['Id']
        ctx_list = item['Context'][-sess_length:]
        ques = item['Question']

        for sent_i in ctx_list:
            if sent_i == '':
                continue

            sent_i_type = sent_i[0]
            sent_i = sent_i[2:].strip()
            sent_i_list = sent_i.split('|||')

            for sent_j in sent_i_list:

                if sent_j.endswith('.jpg'):
                    img_list.append(sent_j)
                    sent_j = '<img>'
                else:
                    img_list.append('NULL')
                    sent_j = clean_text(sent_j)

                sent_seg = ' '.join(tokenize_spt(sent_j.strip()))

                if sent_seg:
                    src_str = src_str + sent_seg + '</s>'
                else:
                    img_list.pop(-1)

        ques_list = ques.split('|||')

        for sent in ques_list:
            if sent.endswith('.jpg'):
                img_list.append(sent)
                sent = '<img>'
            else:
                img_list.append('NULL')
                sent = clean_text(sent)

            sent = sent.strip()
            if sent:
                sent_seg = ' '.join(tokenize_spt(sent.strip()))
                src_str = src_str + sent_seg + '</s>'
            else:
                img_list.pop(-1)

        src_str = src_str[:-4]
        img_str = ' '.join(img_list)

        src_list = src_str.split('</s>')

        assert len(src_list) == len(img_list)

        # lazy to reuse train dataloader code
        trg_str = 'NULL'
        f_out.write(src_str + '\t' + trg_str + '\t' + img_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process online test data")

    parser.add_argument('-d', '--directory', default='online_test_data/')
    parser.add_argument('-t', '--sess_turns', default=2)
    parser.add_argument('-s', '--max_sentence_length', type=int, default=50)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length

    # get the online test file to generate internal data format
    gen_test_set(args.directory, args.sess_turns)

    print("Loading conversations...")
    test, test_img = load_conversations(datasets_dir.joinpath("test.txt"))

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    vocab = Vocab(lang="zh")
    for split_type, conversations, images in [('test', test, test_img)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = datasets_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)
        conversation_length = [min(len(conv), max_conv_len)
                               for conv in conversations]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        images_dir = images_test_dir

        images, images_length = images_str_2_list(images_dir, images, max_conv_length=max_conv_len)
        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))
        to_pickle(images, split_data_dir.joinpath('images.pkl'))
        to_pickle(images_length, split_data_dir.joinpath('images_length.pkl'))

    print('Done!')




