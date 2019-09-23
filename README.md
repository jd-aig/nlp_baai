## Introduction
For better training models or completing tasks in the e-commercial domain, we provide the pre-trained BERT and word embedding. The charts below shows the data we use.

| Task| Data Source | Sentences | Sentence Pairs |
| -- | :--: | :--: | :--: |
| BERT Pre-Training | Customer Service Dialogue Data(CSDD)| 1.2B | - |
| FAQ | LCQMC、CSDD | - | 0.88M for fine-tuning, 80K for test |

| Task | Data Source | Tokens | Vocabulary Size |
| -- | :--: | :--: | :--: |
| Word Embedding Pre-Training | CSDD | 9B | 1M | 
## Download
The links to the models are here.

| Model | Data Source| Link |
| -- | :--: | :--: |
| BAAI-JDAI-BERT, chinese | CSDD | [JD-BERT for Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/models/nlp_baai/20190918/JDAI-BERT.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1600048798&Signature=vv36ssU2iqVasPOdYuBCWIDm5X4%3D)|
| BAAI-JDAI-WordEmbedding | CSDD | [JD-WORD-EMBEDDING with 300d](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/models/nlp_baai/20190918/JDAI-WORD-EMBEDDING.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1600048776&Signature=14rM5LFQywsWHLXhlhGEQAHEE%2FQ%3D)|

```
JD-BERT.tar.gz file contains items:
  |—— BAAI-JDAI-BERT
      |—— bert_model.cpkt.*                  # pre-trained weights
      |—— bert_config.json                   # hyperparamters of the model
      |—— vocab.txt                          # vocabulary for WordPiece
      |—— JDAI-BERT.md & INTRO.md            # summary and details
JD-WORD-EMBEDDING.tar.gz file contains items:
  |—— BAAI-JDAI-WORD-EMBEDDING
      |—— JDAI-Word-Embedding.txt            # word vectors, each line separated by whitespace
      |—— JDAI-WORD-EMBEDDING.md & INTRO.md  # summary and details
```
## BERT
### Pre-training Settings
| Masking | Dataset | Sentences | Training Steps | Device | Init Checkpoint | Init Lr |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| WordPiece | CSDD | 1.2B | 1M | P40×4 | BERT<sup>Google</sup> weight | 1e-4 |
* When pre-training, we do the data preprocessing with our preprocessor including some generalization processing.
* The init checkpoint we use is <12-layer, 768-hidden, 12-heads, 110M parameters>, and bert_config.json and vocab.txt are identical to Google's original settings. 
### Fine-tuning
We use train data of LCQMC, QQ dataset and QA dataset for fine-tuning, and then just train 2 epoches with a proper init learning rate 2e-5 on each dataset respectively. QQ dataset and QA dataset are extracted from other CSDD.
### Evaluation
We evaluate our pre-trained model on the FAQ task with test data of LCQMC, QQ dataset and QA dataset.

#### FAQ
| Model | LCQMC | QQ Test | QA Test|
| -- | :--: | :--: | :--: |
| BERT-wwm | **88.7** | 80.9 | 86.6 |
| Our BERT | 88.6 | **81.9** | **87.5** |

``LCQMC`，```QQ Test`` and ``QA Test`` are the test data containing 5k question pairs, 21k question pairs and 54k question&answer pairs respectively.

## Word Embedding
### Pre-training Settings
| Window Size | Dynamic Window | Sub-sampling | Low-frequency Word | Iter | Negative Sampling<sup>for SGNS</sup> | Dim |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 5 | Yes | 1e-5 | 10 | 10 | 5 | 300 |
* When pre-training, we use our tools for preprocessing and word segmentaion.
* We train the vectors based on Skip-Gram.
### Evaluation
We show top3 similar words for some sample words below. We use cosine distance to compute the distance of two words.

| Input Word | 口红 | 西屋 | 花花公子 | 蓝月亮 | 联想 | 骆驼 |
| -- | :--: | :--: | :--: | :--: | :--: | :--: |
| Similar 1 | 唇釉 | 典泛 | PLAYBOY | 威露士 | 宏碁 | CAMEL |
| Similar 2 | 唇膏 | 法格 | 富贵鸟 | 增艳型 | 15IKB | 骆驼牌 |
| Similar 3 | 纪梵希 | HS1250 | 霸王车 | 奥妙 | 14IKB | 健足乐 |
