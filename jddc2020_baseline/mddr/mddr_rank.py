# -*- coding: utf-8 -*-

import argparse
import json


def gen_answer_result(text_recall_f, text_predict_f, safe_answer_f):

    text_recall = []
    with open(text_recall_f) as f:
        lines = f.readlines()

    for line in lines:
        text_recall.append(json.loads(line.strip()))

    with open(text_predict_f) as f:
        lines = f.readlines()

    text_predict = []
    for line in lines:
        text_predict.append(json.loads(line.strip()))

    ans_result = dict()

    for recall, rank in zip(text_recall, text_predict):
        print(recall)
        print(rank)

        if rank['label'] == '0':
            continue
        else:
            if recall['id'] not in ans_result:
                ans_result[recall['id']] = recall['sentence2']
            else:
                # print('already have high priority answer')
                continue

    for recall in text_recall:
        if recall['id'] not in ans_result:
            ans_result[recall['id']] = recall['sentence2']
        else:
            # print('already have high priority answer')
            continue

    test_answers = list()
    for key in ans_result.keys():
        id = key
        answer = ans_result[key]
        test_answers.append({'Id': id, 'Answer': answer})

    with open(safe_answer_f) as f:
        lines = f.readlines()

    for line in lines:
        safe_ans = (json.loads(line.strip()))
        test_answers.append({'Id': safe_ans['id'], 'Answer': safe_ans['sentence2']})

    with open('./online_test_data/test_answers.json', 'w') as f:
        result = json.dumps(test_answers, ensure_ascii=False, indent=2)
        f.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a rank tool for multi-modal dialogue")
    parser.add_argument('-r', '--recall_text_file', default='./test_text_recall.json')
    parser.add_argument('-n', '--rank_text_file', default='./transformers/jddc_output/bert/test_prediction.json')
    parser.add_argument('-s', '--safe_answer_file', default='./test_safe_answer.json')

    args = parser.parse_args()

    gen_answer_result(args.recall_text_file, args.rank_text_file, args.safe_answer_file)
