# -*- coding: utf-8 -*-

import argparse
import re
import jieba
import fasttext
import random

random.seed(5)

chat_class_model = 'chat/data_dim256_lr00.5_iter20.model'
chat_label = '__label__1'

pattern_pun = '！，；：？、。"!,;:?."\''
pattern_jpg = re.compile(r'[A-Za-z0-9]+\.jpg')


def clean_text(text):

    # text = re.sub(r'[{}]+'.format(r'\d+\*\*'), '<num>', text)
    # text = re.sub(r'[{}]+'.format(r'\d+'), '<num>', text)

    # text = clean_punctuation(text)
    return text


def clean_punctuation(text):
    text = re.sub(r'[{}]+'.format(pattern_pun), '', text)
    return text.strip().lower()


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


class DataIterm(object):
    def __init__(self, sid, ques, ans, ctx):
        self.sid = sid
        self.ques = ques
        self.ans = ans
        self.ctx = ctx


def do_preprocess(dir, sess_turn):
    """
    :param dir:  官方数据存放路径
    :param sess_turn: context中保存的历史上下文的对话轮数
    :return: train_items, dev_items
             用于训练的train和dev数据，其中每条数据记录由以下几部分原始信息组成
             sid, 对话原始的session信息，后续按照需要可以根据该信息查询对话相关的知识库，本实例中未使用
             question, 该条训练数据所对应的用户的问题
             answer, 该条训练数据对应的客服的回答
             context, 该对话发生的上下文信息，该信息最大信息长度不超过sess_turn所定义的轮数
    """
    sess_len = sess_turn * 2
    train_items = list()
    dev_items = list()

    for file, item_list in [('data_train.txt', train_items), ('data_dev.txt', dev_items)]:
        with open(dir+file) as f:
            lines = f.readlines()

        data_list = list()
        sess_pid = dict()
        for line in lines:
            word = line.strip().split('\t')
            sid = word[0]
            shop = word[1]
            pid = word[2]
            text = word[3]
            waiter = word[4]

            if pid:
                sess_pid[sid] = pid

            if waiter == '1':
                text = 'A:' + text
            else:
                text = 'Q:' + text

            data_list.append((sid, text))

        data_len = len(data_list)
        i = 0

        tmp_data_list = list()

        # 将原始数据按照session和问题、回答类型，
        # 用'|||'连接不同回车发送的内容
        while i < data_len:

            i_head = data_list[i][1][0]
            i_text = data_list[i][1]
            i_sid = data_list[i][0]

            j = i+1
            if j >= data_len:
                tmp_data_list.append((i_sid, i_text))
                break

            j_head = data_list[j][1][0]
            j_text = data_list[j][1]
            j_sid = data_list[j][0]

            add = 0
            while i_head == j_head and i_sid == j_sid:
                i_text = i_text + '|||' + j_text[2:]
                add = add + 1
                j = j + 1

                if j >= data_len:
                    break

                j_head = data_list[j][1][0]
                j_text = data_list[j][1]
                j_sid = data_list[j][0]

            i = i + add + 1
            tmp_data_list.append((i_sid, i_text))

        # 遍历全部（session, Q:xxx） (session, A:xxx),
        # 构建训练输入文件，Q，A，Context，
        # 其中'@@@'间隔Context里面不同的Q或者A
        for idx, item in enumerate(tmp_data_list):

            sid = item[0]
            text = item[1]

            if text.startswith('A'):
                continue

            question = text.replace('Q:', '').strip()

            if question == '':
                continue

            if idx+1 >= len(tmp_data_list):
                continue

            n_item = tmp_data_list[idx+1]
            n_sid = n_item[0]

            if sid != n_sid:
                continue

            n_text = n_item[1]

            answer = n_text.replace('A:', '').strip()

            if answer == '':
                continue

            if idx > sess_len:
                cand_data_list = tmp_data_list[idx-sess_len:idx]
            else:
                cand_data_list = tmp_data_list[:idx]

            contxt_list = list()
            for cand_item in cand_data_list:
                cand_sid = cand_item[0]
                cand_text = cand_item[1]

                if cand_sid != sid:
                    continue
                contxt_list.append(cand_text)

            context = '@@@'.join(contxt_list)

            item_list.append(DataIterm(sid, question, answer, context))

    return train_items, dev_items


def clean_qa_data(input_text):
    resp = list()
    sents = input_text.split('|||')
    for sent in sents:

        if sent in ['用户发起转人工', '售后咨询组', '人工服务', '售前咨询组', '我要转人工', '未购买→售前咨询组', '已购买→售后咨询组']:
            continue

        if sent.endswith('jpg'):
            continue
        else:
            sent = clean_text(sent)

        resp.append(sent)
    return ' '.join(resp)


def is_discard_ques(classifier, ques):
    if ques in ['', '<url>', '<num>']:
        return True
    if len(ques) < 4:
        return True

    if '谢谢' in ques:
        return True

    if '购买前咨询' in ques:
        return True

    ques_char = [i for i in ques.replace(' ', '')]
    ques_pred = ' '.join(ques_char)

    pred = classifier.predict(ques_pred)

    pre_label, sim = pred[0][0], pred[1][0]

    if pre_label == chat_label:
        return True

    return False


def is_discard_ans(ans):
    if len(ans) < 4:
        return True

    for item in ['欢迎光临', '欢迎小主光临', '稍等', '有什么可以帮您']:
        if item in ans:
            return True

    if ans in ['<url>', '<num>', '#E-s<num>']:
        return True

    return False


def get_cand_qa(items):

    f_out = open('QA_dbs.txt', 'w')
    classifier = fasttext.load_model(chat_class_model)

    for item in items:
        ques = clean_qa_data(item.ques).strip()
        ans = clean_qa_data(item.ans).strip()

        if is_discard_ques(classifier, ques) or is_discard_ans(ans):
            continue

        f_out.write(ques+'\t'+ans+'\n')


def clean_img_qa_data(input):

    resp = list()
    sents = input.split('|||')
    img_list = list()

    for sent in sents:

        if sent in ['用户发起转人工', '售后咨询组', '人工服务', '售前咨询组', '我要转人工', '未购买→售前咨询组', '已购买→售后咨询组']:
            continue

        if sent.endswith('jpg'):
            img_list.append(sent)
            continue
        else:
            sent = clean_text(sent)

        resp.append(sent)

    return ' '.join(resp).strip(), img_list


def get_cand_img_qa(items):

    f_out = open('img_QA_dbs.txt', 'w')

    for item in items:
        ques, ques_images = clean_img_qa_data(item.ques)
        ans, ans_images = clean_img_qa_data(item.ans)

        if not ques_images:
            continue
        if ans_images:
            continue
        if is_discard_ans(ans):
            continue

        for img in ques_images:
            f_out.write(img + '\t' + ans + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process raw data")

    parser.add_argument('-d', '--directory', default='data/')
    parser.add_argument('-s', '--sess_turns', default=2)

    args = parser.parse_args()

    train_items, dev_items = do_preprocess(args.directory, args.sess_turns)

    get_cand_qa(train_items)
    get_cand_img_qa(train_items)
