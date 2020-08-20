import json
import pickle


def postprocess():
    """
    从result.txt文件中读出模型预测结果，
    与id_list一起对应生成test_answer.json
    """
    res_list = []
    with open('./data/test/id_list.pkl', 'rb') as f:
        id_list = pickle.load(f)

    f = open('result.txt', 'r').readlines()

    assert len(id_list) == len(f)

    for id, answer in zip(id_list, f):
        res_dict = {"Id": id, "Answer": answer.strip()}
        res_list.append(res_dict)

    with open('online_test_data/test_answer.json', 'w') as f:
        data = json.dumps(res_list, ensure_ascii=False, indent=2)
        f.write(data)


if __name__ == '__main__':
    postprocess()


