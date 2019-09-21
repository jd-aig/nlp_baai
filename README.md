## Guide
| Section | About |
| -- | -- |
| [Introduction](#Introduction) | The dataset we used in the whole experiment|
| [Download](#Download) | Download links for the pre-trained BERT & Word Embedding|
| [BERT](#BERT) | Details of pre-training and results on two evaluation tasks |
| [Word Embedding](#Word-Embedding)| Details pre-training and some evaluation results |
## Download
| Model | Data Source| Link |
| -- | -- | -- |
| JD BERT, chinese | CSDD | [Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/models/nlp_baai/20190918/JDAI-BERT.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1600048798&Signature=vv36ssU2iqVasPOdYuBCWIDm5X4%3D)|
| JD WordEmbedding | CSDD | [300d](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/models/nlp_baai/20190918/JDAI-WORD-EMBEDDING.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1600048776&Signature=14rM5LFQywsWHLXhlhGEQAHEE%2FQ%3D)|
## Introduction
For better training models or completing tasks in the e-commercial domain, we provide pre-trained BERT and word embedding. The charts below shows the data we used.

| Task| Data Source | Sentences | Sentence Pairs |
| -- | -- | -- | -- |
| BERT Pre-Training | Customer Service Dialogue Data(CSDD)| 1.2B | - |
| QAP | - | - | 0.43M for train, 54K for test |
| QQC | - | - | 0.47M for train, 25K for test |

| | Data Source |Tokens | Vocabulary Size |
| -- | -- | -- | -- |
| Word Embedding Pre-Training | CSDD | 9B | 1M | 
## BERT
### Pretrain Settings
| Masking | Dataset | Sentences | Training Steps | Device | Init Checkpoint | Init Lr |
| -- | -- | -- | -- | -- | -- | -- |
| WordPiece | Customer Service Dialogue Data | 1.2B | 1M | P40×4 | BERT<sup>Google</sup> weight | 1e-4 |
* When pre-training, we do the data preprocessing with our preprocessor including some generalization processing.
* The init checkpoint we use is <12-layer, 768-hidden, 12-heads, 110M parameters>, and bert_config.json and vocab.txt are identical to Google's original settings. 
### Finetune
We feed the train data into the model, and just train 2 epoches with a proper init learning rate 2e-5.
### Evaluation
We evaluate our pre-trained model on QA Pairs and QQ classificaiton tasks. In the QQ classification task, we do evaluation on both LCQMC data and our test data to compare the model's performance on e-commercial data with on data in the general domain.
#### QA Pairs
| Model | Test |
| -- | -- |
| BERT-wwm | 86.6 |
| **Our BERT** | **87.5** |
#### QQ Classification
| Model | LCQMC Test |
| -- | -- |
| **BERT-wwm** | **88.7** |
| Our BERT | 88.6 |

| Model | Test |
| -- | -- |
| BERT-wwm | 80.9 |
| **Our BERT** | **81.9** |
## Word Embedding
### Pretrain Settings
| Window Size | Dynamic Window | Sub-sampling | Low-frequency Word | Iter | Negative Sampling<sup>for SGNS</sup> | Dim |
| -- | -- | -- | -- | -- | -- | -- |
| 5 | Yes | 1e-5 | 10 | 10 | 5 | 300 |
* When pre-training, we use our tools for preprocessing and word segmentaion.
* We train the vectors based on Skip-Gram.
### Evaluation
We show top3 similar words for some sample words below. We use cosine distance to compute the distance of two words.

| Input Word | 口红 | 西屋 | 花花公子 | 蓝月亮 | 联想 | 骆驼 |
| -- | -- | -- | -- | --| -- | -- |
| Similar 1 | 唇釉 | 典泛 | PLAYBOY | 威露士 | 宏碁 | CAMEL |
| Similar 2 | 唇膏 | 法格 | 富贵鸟 | 增艳型 | 15IKB | 骆驼牌 |
| Similar 3 | 纪梵希 | HS1250 | 霸王车 | 奥妙 | 14IKB | 健足乐 |
