# nlp_baai
1. JD AI Lab embedding trained on JD Customer Service Dialogue Data
2. JD AI Lab Chinese BERT trained on JD Customer Service Dialogue Data
## Guide
| Section | About |
| -- | -- |
| [Introduction](#Introduction) | show the dataset we used in the whole expriment|
| [Download](#Download) | where to download the model & word vectors|
| [BERT](#BERT) | including pretraining and results on two evaluation tasks |
| [Word Embedding](#Word-Embedding)| including pretraining and results on two evaluation tasks |
## Download
| Model | Data | Link |
| -- | -- | -- |
| JD BERT, chinese | CSDA | [Tensorflow](https://...)|
| JD WordEmbedding | CSDA | [300d](https://...)|
## Introduction
For better training models or completing tasks in the e-commercial domain, we provide pre-trained BERT and word embedding. The charts below shows the data we used.

| | token | pairs |
| -- | -- | -- |
| BERT | 9B | - |
| QAP | - | 0.43M for train, 54K for test |
| QQC | - | 0.47M for train, 25K for test |

| | token | vocab | pairs |
| -- | -- | -- | -- |
| word embedding | 9B | 0.1B | - |
| analogy task | - | - | 500 |
| QQ similarity | - | - | 500 |
## BERT
### Pretrain Settings
| masking | dataset | token | Training Steps | Device | init checkpoint | init lr |
| -- | -- | -- | -- | -- | -- | -- |
| WordPiece | Customer Service Dialogue Data | 9B | 1M | P40 | BERT<sup>Google</sup> weight | 1e-4 |
* When pre-training, we do the data preprocessing with our preprocessor including some generalization processing.
* The init checkpoint we use is <12-layer, 768-hidden, 12-heads, 110M parameters>, and bert_config.json and vocab.txt are identical to Google's original settings. 
### Finetune
We feed the train data into the model, and just train 2 epoches with a proper init learning rate 2e-5.
### Evaluation
We evaluate our pre-trained model on QA Pairs and QQ classificaiton tasks. In the QQ classification task, we do evaluation on both LCQMC data and our test data to compare the model's performance on e-commercial data with on data in the general domain.
#### QA Pairs
| Model | test |
| -- | -- |
| BERT-wwm | 86.6 |
| **Our BERT** | **87.5** |
#### QQ Classification
| Model | LCQMC test |
| -- | -- |
| **BERT-wwm** | **88.7** |
| Our BERT | 88.6 |

| Model | test |
| -- | -- |
| BERT-wwm | 80.9 |
| **Our BERT** | **81.9** |
## Word Embedding
### Pretrain Settings
| window size | dynamic window | sub-sampling | low-frequency word | iter | negative sampling<sup>for SGNS</sup> | dim |
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
