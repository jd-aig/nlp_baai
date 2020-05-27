import random
import json


class Item(object):
    def __init__(self, label, sentence1, sentence2):
        self.label = label
        self.sentence1 = sentence1
        self.sentence2 = sentence2


def get_all_qa_pairs(file):
    with open(file) as f:
        lines = f.readlines()

    pos_qa = list()
    neg_qa = list()

    for line in lines:
        word = line.strip().split('\t')
        pos_qa.append({'label': '1', 'sentence1': word[0], 'sentence2': word[1]})

    len_pos_qa = len(pos_qa)

    for item in pos_qa:
        ques = item['sentence1']
        idx = random.randint(0, (len_pos_qa-1))
        ans = pos_qa[idx]['sentence2']
        neg_qa.append({'label': '0', 'sentence1': ques, 'sentence2': ans})

    all_qa = pos_qa + neg_qa
    random.shuffle(all_qa)

    qa_len = len(all_qa)
    qa_len_08 = int(qa_len * 0.8)
    qa_len_09 = int(qa_len * 0.9)

    qa_train = all_qa[:qa_len_08]
    qa_dev = all_qa[qa_len_08:qa_len_09]
    qa_test = all_qa[qa_len_09:]

    f_train = open('./data/jddc/train.json', 'w')
    for item in qa_train:
        f_train.write(json.dumps(item, ensure_ascii=False)+'\n')

    f_dev = open('./data/jddc/dev.json', 'w')
    for item in qa_dev:
        f_dev.write(json.dumps(item, ensure_ascii=False) + '\n')

    f_test = open('./data/jddc/test.json', 'w')
    for item in qa_test:
        f_test.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    get_all_qa_pairs('../QA_dbs.txt')


if __name__ == '__main__':
    main()