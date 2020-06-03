# -*- coding: utf-8 -*-
import argparse
import jieba
from gensim.models import KeyedVectors
import numpy as np
import argparse
from indexers.faiss_indexers import *
from tqdm import tqdm
from utils.doc import *
import json
import fasttext
import PIL
import torch
import torchvision
from torchvision import transforms


chat_label = '__label__1'

data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_text_embedding_idx(idx_f, qa_f, w2v_f):

    w2v = KeyedVectors.load_word2vec_format(w2v_f)

    index = DenseFlatIndexer(200)
    index.deserialize_from(idx_f)

    with open(qa_f) as f:
        lines = f.readlines()

    document = []

    for line in lines:
        word = line.strip().split('\t')
        ques = word[0]
        ans = word[1]

        document.append((ques, ans))

    return w2v, index, document


def get_img_embedding_idx(idx_f, qa_f):

    index = DenseFlatIndexer(512)
    index.deserialize_from(idx_f)

    with open(qa_f) as f:
        lines = f.readlines()

    document = []

    for line in lines:
        word = line.strip().split('\t')
        ques = word[0]
        ans = word[1]

        document.append((ques, ans))

    model_ft = torchvision.models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    feature_extractor.eval()

    return feature_extractor, index, document


def get_cand_ques_doc(query, w2v, idx, document, best_num):
    q_emb = get_sentence_emb(w2v, query)
    q = q_emb[np.newaxis, :]

    result = idx.search_knn(q, best_num)
    I,D = result[0]
    cand_docs = list()
    for idx in I:
        cand_docs.append(document[idx])

    return cand_docs


def get_last_text_question(ctx):
    ques_list = list()
    for item in ctx:
        if item.startswith('Q:'):
            item = item.replace('Q:', '')
            ques_list.append(item)
    if ques_list:
        last_q = ques_list[-1]
    else:
        last_q = ''
    return last_q


def get_clean_text_question(text):
    sents = text.split('|||')

    sent_list = list()
    for sent in sents:
        if sent in ['', '<url>', '<num>']:
            continue
        if '购买前咨询' in sent:
            continue
        if '售后咨询组' in sent:
            continue
        if sent.endswith('.jpg'):
            continue
        sent_list.append(sent)

    return ' '.join(sent_list)


def get_valid_img_ques(context, question):
    img_list = list()

    sents = question.strip().split('|||')

    for sent in sents:
        if sent.endswith('.jpg'):
            img_list.append(sent)

    if not img_list:
        last_ques = get_last_text_question(context)

        sents = last_ques.strip().split('|||')

        for sent in sents:
            if sent.endswith('.jpg'):
                img_list.append(sent)

    return img_list


def get_img_input(file):
    img_tmp = PIL.Image.open(file)
    img = data_transforms(img_tmp)
    return img


def get_cand_img_ques_doc(img_f, f_e, index, document):

    img_in = get_img_input(img_f)
    img_in = img_in[np.newaxis, :]
    img_ft = f_e(img_in).data.numpy().squeeze()
    img_ft = img_ft[np.newaxis, :]

    result = index.search_knn(img_ft, 1)
    I, D = result[0]
    cand_docs = list()
    for idx in I:
        cand_docs.append(document[idx])

    return cand_docs


def main(w2v_f, qa_f, idx_f, ft_m, qa_img_f, idx_img_f):

    classifier = fasttext.load_model(ft_m)
    w2v, index, document = get_text_embedding_idx(idx_f, qa_f, w2v_f)
    f_e, index_img, document_img = get_img_embedding_idx(idx_img_f, qa_img_f)

    with open('online_test_data/test_questions.json') as f:
        questions = json.load(f)

    f_out_recall = open('test_text_recall.json', 'w')
    f_out_safe = open('test_safe_answer.json', 'w')
    for question in questions:
        id = question['Id']
        context = question['Context']
        question = question['Question']

        ques = get_clean_text_question(question)

        if ques:
            ques_chars = ' '.join(ques).strip()
            pred_res = classifier.predict(ques_chars)
            if pred_res[0][0] == chat_label:
                last_ques = get_last_text_question(context)
                last_ques = get_clean_text_question(last_ques)
                ques = last_ques
        else:
            last_ques = get_last_text_question(context)
            last_ques = get_clean_text_question(last_ques)
            ques = last_ques

        if ques:
            resp = get_cand_ques_doc(ques, w2v, index, document, 20)
            print(resp)

            for item in resp:
                f_out_recall.write(json.dumps({'id': id, 'sentence1': ques, 'sentence2': item[1]},
                                              ensure_ascii=False)+'\n')
        else:
            # when no chance to recall by text, try to recall by image or give a safe answer directly
            img_list = get_valid_img_ques(context, question)

            if img_list:
                img = './online_test_data/images_test/'+img_list[0]

                resp = get_cand_img_ques_doc(img, f_e, index_img, document_img)

                for item in resp:
                    f_out_safe.write(json.dumps({'id': id, 'sentence1': question, 'sentence2': item[1]},
                                                ensure_ascii=False)+'\n')
            else:
                f_out_safe.write(json.dumps({'id': id, 'sentence1': question, 'sentence2': '嗯嗯~'},
                                            ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a retriever tool for multi-modal dialogue")
    parser.add_argument('-w', '--w2v_file', default='1000000-small.txt')
    parser.add_argument('-q', '--qa_file', default='QA_dbs.txt')
    parser.add_argument('-a', '--img_qa_file', default='img_QA_dbs.txt')
    parser.add_argument('-i', '--index_file', default='jddc')
    parser.add_argument('-n', '--img_index_file', default='jddc_img')
    parser.add_argument('-f', '--fasttext_model', default='./chat/data_dim256_lr00.5_iter20.model')

    args = parser.parse_args()

    main(args.w2v_file, args.qa_file, args.index_file, args.fasttext_model, args.img_qa_file, args.img_index_file)
