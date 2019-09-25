## Introduction

This is project for releasing some open-source natural language models from Joint Lab of [BAAI](https://www.baai.ac.cn/) and [JDAI](http://air.jd.com/).
Different from other open-source Chinese NLP models, we mainly focus on some basic models for dialogue systems, especially in E-commerce domain.
Our corpus is very huge, currently we are using 42 GB Customer Service Dialogue Data (CSDD) for training, and it contain about **1.2 billion** sentences.

We provide the pre-trained BERT and word embeddings. The charts below shows the data we use.

| Task| Data Source | Sentences | Tokens | Vocabulary Size |
| -- | :--: | :--: | :--: | :--: |
| Pre-Training | CSDD(Customer Service Dialog Data)| 1.2B | 9B | 1M |

## Download
The links to the models are here.

| Model | Data Source| Link |
| -- | :--: | :--: |
| BAAI-JDAI-BERT, Chinese | CSDD | [JD-BERT for Tensorflow](https://jdai009.s3.cn-north-1.jdcloud-oss.com/jd-aig/open/models/nlp_baai/20190918/JDAI-BERT.tar.gz?AWSAccessKeyId=BB50A587AB371E21919040C802767A0C&Expires=1600048798&Signature=vv36ssU2iqVasPOdYuBCWIDm5X4%3D)|
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
* The init checkpoint we use is ``<12-layer, 768-hidden, 12-heads, 110M parameters>, Chinese``, and ``bert_config.json`` and ``vocab.txt`` are identical to Google's original settings. 
* We do not use ``Chinese Whole Word Masking(WWM)`` in our current pre-training but use google's original WWM which is on the Chinese character level.
### Fine-tuning
We use ``Train`` data of LCQMC(Large-scale Chinese Question Matching Corpus) and CSDQMC(Customer Service Dialog Question Matching Corpus) for fine-tuning, and then just train 2 epochs with a proper init learning rate 2e-5 on each dataset respectively. 

| Dataset | Train | Test | Domain | MaxLen | Batch Size | Epoch |
| -- | :--: | :--: | :--: | :--: | :--: | :--: |
| LCQMC | 140K | 12.5K | Zhidao | 128 | 32 | 2 |
| CSDQMC | 200K | 9K | Customer Service | 128 | 32 | 2 |

### Evaluation
We evaluate our pre-trained model on the FAQ task with ``Test`` data of LCQMC and CSDQMC.
#### FAQ Task
| Model | LCQMC | CSDQMC |
| -- | :--: | :--: |
| ERNIE |87.2|-|
| BERT |86.9|85.1|
| BERT-wwm | **88.7** | 86.6 |
| BAAI-JDAI-BERT | 88.6 | **87.5** |

We quote the ``BERT`` and ``ERNIE``'s results on LCQMC from the [`Chinese-BERT-wwm report`](https://arxiv.org/pdf/1906.08101.pdf).
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
