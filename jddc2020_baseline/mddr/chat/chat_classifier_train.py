import pandas
import fasttext
import numpy as np
import os
from sklearn.metrics import classification_report


def data_train_test():
    f_train = open('train.txt', 'w')
    f_test = open('test.txt', 'w')

    df_train = pandas.read_excel('chat.xlsx', sheetname='train')
    df_train.fillna(0, inplace=True)
    df_test = pandas.read_excel('chat.xlsx', sheetname='test')
    df_test.fillna(0, inplace=True)

    for i in range(0, len(df_train)):
        row = df_train.iloc[i]
        text  = row['text']
        label = row['label']

        text = [text[i] for i in range(0, len(text))]
        line = '__label__'+str(int(label))
        # print(line+' '+' '.join(text))
        f_train.write(line+' '+' '.join(text)+'\n')

    for i in range(0, len(df_test)):
        row = df_test.iloc[i]
        text = row[0]
        label = row[1]

        text = [text[i] for i in range(0, len(text))]
        line = '__label__' + str(int(label))
        f_test.write(line + ' ' + ' '.join(text)+'\n')


def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=2, loss=loss)
        """
          训练一个监督模型, 返回一个模型对象

          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier


def cal_precision_and_recall(classifier, file='test.txt'):
    with open(file) as f:
        lines = f.readlines()

    y_truth = list()
    y_pred = list()

    for line in lines:
        word = line.split(' ')
        lable = word[0]
        text = ' '.join(word[1:]).strip()

        pred_res = classifier.predict(text)

        pre_label, sim = pred_res[0][0], pred_res[1][0]

        y_truth.append(int(lable.replace('__label__', '')))
        y_pred.append(int(pre_label.replace('__label__', '')))

    print(classification_report(y_truth, y_pred, target_names=['class 0', 'class 1']))


def main():
    data_train_test()

    dim = 256
    lr = 0.5
    epoch = 20
    model = f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

    classifier = train_model(ipt='train.txt',
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=lr
                             )

    result = classifier.test('test.txt')
    print(result)

    cal_precision_and_recall(classifier, 'test.txt')


if __name__ == '__main__':
    main()