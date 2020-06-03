import json


def online_test_data_output():

    output_list = list()
    # load question id
    with open('./online_test_data/test_questions.json') as f:
        ques_items = json.load(f)

    with open('./pred/res.txt') as f:
        predictions = f.readlines()

    for question, answer in zip(ques_items, predictions):
        result_dict = dict()

        result_dict['Id'] = question['Id']
        answer = answer.strip().split('\t')[2]
        result_dict['Answer'] = answer
        output_list.append(result_dict)

    with open('./online_test_data/test_answers.json', 'w') as f:
        result = json.dumps(output_list, ensure_ascii=False, indent=2)
        f.write(result)


if __name__ == '__main__':
    online_test_data_output()