## Introduction

This is the open-source project from Joint Lab of [BAAI](https://www.baai.ac.cn/) and [JDAI](http://air.jd.com/).

We release four models for building dialogue systems including Intent Classification (NLU), Slot Tagging (NER), Dialogue State Tracking (DST) and Question & Answering (QA). NLU and QA models are bert-based models. NER model is based on Bi-LSTM + CRF enhanced by some linguistic features. DST model is the [TRADE](https://arxiv.org/pdf/1905.08743.pdf) model. The former three models are trained on our real Customer Service Dialogue Data (CSDD) and the last DST model is trained on public dataset [CrossWOZ](https://arxiv.org/abs/2002.11893).

## Download

The download links of these models are as below.

| Model                            | Data Source |                             Link                             |
| -------------------------------- | :---------: | :----------------------------------------------------------: |
| BAAI-JD-Dialogue-Intent, Chinese |    CSDD     | [BAAI-JD-Dialogue-Intent for Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/nlp_baai/20200819/BAAI-JD-Dialogue-Intent.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1660027347&Signature=sfmUMhfxL5jTx2LptiTzfqMuXwo%3D) |
| BAAI-JD-Dialogue-Ner, Chinese    |    CSDD     | [BAAI-JD-Dialogue-Ner for Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/nlp_baai/20200819/BAAI-JD-Dialogue-Tagging.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1660027369&Signature=5xB9ADooIA%2Faup41HiTxXUqIU4o%3D) |
| BAAI-JD-Dialogue-Dst, Chinese    |  CrossWOZ   | [BAAI-JD-Dialogue-Dst for Pytorch](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/nlp_baai/20200819/BAAI-JD-Dialogue-DST.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1660027322&Signature=X0Vl7aufHl4yqbfOW6Ve3dUe72M%3D) |
| BAAI-JD-Dialogue-Sim, Chinese    |    CSDD     | [BAAI-JD-Dialogue-Sim for Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/nlp_baai/20200819/BAAI-JD-Dialogue-Sim.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1660027359&Signature=QimcWublc8fNeQEpXuYig3zG6lE%3D) |

## Performance

**1. BAAI-JD Dialogue Intent** 

This model is used for intent classification in dialogue system.
We define 10 intents in this task, which are the most common intents in E-commerce customer service domain. The model is trained on BAAI-JDAI-BERT, by fine-tuning with in-house annotated intent classification corpus, and the max_seq_length is set to 50. 

| Intent           | Precision | Recall | F1    | Support |
| ---------------- | --------- | ------ | ----- | ------- |
| 配送周期         | 93.11     | 96.83  | 94.93 | 600     |
| 什么时间出库     | 97.44     | 95.00  | 96.20 | 600     |
| 售后商品使用问题 | 96.48     | 95.83  | 96.15 | 600     |
| 商品区别         | 97.70     | 99.17  | 98.43 | 600     |
| 修改订单         | 98.16     | 98.00  | 98.08 | 600     |
| 商品推荐         | 96.69     | 97.33  | 97.01 | 600     |
| 赠品             | 96.70     | 97.67  | 97.18 | 600     |
| 能否便宜优惠     | 95.84     | 96.00  | 95.92 | 600     |
| 家电安装         | 96.89     | 98.50  | 97.69 | 600     |
| 配送方式         | 98.33     | 98.33  | 98.33 | 600     |
| other            | 86.22     | 81.33  | 83.70 | 600     |

|          | Macro Precision | Macro Recall | Macro F1 | Accuracy |
| -------- | ------- | ------- | -------- | -------- |
| 全测试集 | 95.78   | 95.82   | 95.78    | 95.82    |

**2. BAAI-JD Dialogue Tagging**

This model is used for name entity recognition in dialogue system.
We define 6 types of entities in this task, which are also very common in the E-commerce customer service domain. The model is based on Bi-LSTM with CRF, and enhanced by some linguistic features.

| Entity   | Precision | Recall | F1    | Support |
| -------- | --------- | ------ | ----- | ------- |
| brand    | 83.33     | 81.58  | 82.45 | 186     |
| date     | 89.91     | 92.82  | 91.34 | 446     |
| location | 89.94     | 89.94  | 89.94 | 467     |
| price    | 85.67     | 89.67  | 87.62 | 314     |
| product  | 80.44     | 80.34  | 80.39 | 869     |
| time     | 88.15     | 87.67  | 87.91 | 363     |

|          | Macro Precision | Macro Recall | Macro F1 |
| -------- | ------- | ------- | -------- |
| 全测试集 | 85.60   | 86.28   | 85.94    |

**3. BAAI-JD Dialogue DST**

This model is used for dialogue state tracking for multi-domains.
The task is to identify slot-value pairs in the query in multi-turn dailogues, meanwhile maintain the dailogue states. The model is based on TRADE model, and enhanced by the BAAI-JDAI-WordEmbedding model.

|          | Joint Accuracy | Turn Accuracy | Joint F1 |
| -------- | -------------- | ------------- | -------- |
| CrossWOZ | 24.40          | 97.75         | 77.90    |

**4. BAAI-JD Dialogue Sim**

This model is used for calculating the semantic similarity between question and answer, and it can be used for builiding retrieval based dialogue system.
The task has 2 labels, where "0" means "not match" and "1" means "match". The model is based on BAAI-JDAI-BERT, and fine-tuned with in-house annotated QA matching corpus. 
After load the model, you can concatenate the question and answer as "<CLS>Q<SEP>A<SEP>", and the model will predict the similarity score. The max_seq_length is set to 128. 

|          | Macro Precision | Macro Recall | Macro F1 | Accuracy |
| -------- | ------- | ------- | -------- | -------- |
| 全测试集 | 82.16   | 82.15   | 82.15    | 82.15    |

For more details, you can download the [model package](#Download) and refer to the codes and README in it.

