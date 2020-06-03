import json
import pickle
import argparse
import os
from raw_data_process import clean_text
from utils_tf import Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from pathlib import Path


def pad_sentences(conversations, max_sentence_length=50, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        tokens = tokens.strip().split(' ')
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation.reverse()
            conversation = conversation[:max_conversation_length]
            conversation.reverse()

        sentence_length = [min(len(sentence.strip().split()) + 1, max_sentence_length)
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


def clean_contxet(input):
    res = []
    for text in input:
        t = clean_text(text)
        if t is '':
            continue
        if t.endswith('jpg'):
            t = list(t)
            t.insert(-3, '.')
            t = ''.join(t)
        res.append(t)
    return res


def images_str_2_list(images, max_conv_length):
    all_img_list = list()
    all_img_len_list = list()

    for image in images:
        img_list = image.strip().split()

        name_list = ['NULL']*max_conv_length
        mark_list = [0]*max_conv_length
        img_list.reverse()
        for idx, item in enumerate(img_list):
            if item == 'NULL':
                None
            else:
                name_list[idx] = item
                mark_list[idx] = 1
            if idx+1 == max_conv_length:
                break

        name_list = name_list[:max_conv_length-1]
        mark_list = mark_list[:max_conv_length-1]
        name_list.reverse()
        mark_list.reverse()

        all_img_list.append(name_list)
        all_img_len_list.append(mark_list)

    return all_img_list, all_img_len_list


if __name__ == '__main__':
    """
    读取test_question_json文件，在data文件夹下创建test文件夹并生成6个pkl文件
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--max_sentence_length', type=int, default=50)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_vocab_frequency', type=int, default=1)
    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency

    f = open('./online_test_data/test_questions.json.example', 'r')
    q_list = json.load(f)
    f.close()
    id_list = []

    conversations = []
    for q_dict in q_list:
        # 取出一个样本
        id = q_dict['Id']
        id_list.append(id)
        input = []
        context = q_dict['Context']
        question = q_dict['Question']

        for sent in context:
            sent = sent[2:]
            sent = sent.strip().split('|||')
            input.extend(sent)
        input.extend(question.strip().split('|||'))
        input.append('no_target')
        input = clean_contxet(input)
        conversations.append(input)
    conversation_length = [min(len(conv), max_conv_len)
                           for conv in conversations]

    sentence, sentence_length = pad_sentences(conversations)

    images = []
    for conversation in conversations:
        image = ''
        for sent in conversation:
            if sent.endswith('.jpg'):
                image += sent + ' '
            else:
                image += 'NULL' + ' '
        images.append(image.strip())

    images, images_length = images_str_2_list(images, max_conv_length=max_conv_len)

    def show(x):
        print(sentence[:x])
        print(sentence_length[:x])
        print(images[:x])
        print(images_length[:x])
        print(conversation_length[:x])
        print(id_list[:x])

    show(3)

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    data_dir = Path('./data/test')
    data_dir.mkdir(exist_ok=True)

    to_pickle(conversation_length, data_dir.joinpath('conversation_length.pkl'))
    to_pickle(sentence, data_dir.joinpath('sentences.pkl'))
    to_pickle(sentence_length, data_dir.joinpath('sentence_length.pkl'))
    to_pickle(images, data_dir.joinpath('images.pkl'))
    to_pickle(images_length, data_dir.joinpath('images_length.pkl'))
    to_pickle(id_list, data_dir.joinpath('id_list.pkl'))