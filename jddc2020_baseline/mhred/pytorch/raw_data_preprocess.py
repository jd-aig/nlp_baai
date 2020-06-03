# -*- coding: utf-8 -*-

import argparse
import re
import jieba

pattern_pun = '！，；：？、。"!,;:?."\''
pattern_jpg = re.compile(r'[A-Za-z0-9]+\.jpg')


def clean_text(text):

    text = re.sub(r'[{}]+'.format(r'\d+\*\*'), '<num>', text)
    text = re.sub(r'[{}]+'.format(r'\d+'), '<num>', text)

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


def gen_train_dev_set(dir, train_items, dev_items):
    f_train_out = open(dir + 'train' + '.txt', 'w')
    f_dev_out = open(dir + 'dev' + '.txt', 'w')

    for type in ['train', 'dev']:
        if type == 'train':
            items = train_items
            f_out = f_train_out
        elif type == 'dev':
            items = dev_items
            f_out = f_dev_out

        for item in items:

            img_list = list()
            src_str = ''
            trg_str = ''

            ques = item.ques.strip()
            ans = item.ans.strip()
            ctx = item.ctx

            ctx_list = ctx.split('@@@')

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

            ans_list = ans.split('|||')

            for sent in ans_list:
                if sent.endswith('jpg'):
                    sent = '<img>'
                else:
                    sent = clean_text(sent)

                trg_str = trg_str + ' ' + ' '.join(tokenize_spt(sent.strip()))

            src_str = src_str[:-4]
            trg_str = trg_str.strip()
            img_str = ' '.join(img_list)

            src_list = src_str.split('</s>')

            assert len(src_list) == len(img_list)

            if '<img>' not in trg_str:
                f_out.write(src_str+'\t'+trg_str+'\t'+img_str+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process raw data")

    parser.add_argument('-d', '--directory', default='data/')
    parser.add_argument('-s', '--sess_turns', default=2)

    args = parser.parse_args()

    train_items, dev_items = do_preprocess(args.directory, args.sess_turns)

    gen_train_dev_set(args.directory, train_items, dev_items)
