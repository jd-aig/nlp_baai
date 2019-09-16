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
| masking | dataset | token | Training Steps | Device | init checkpoint |
| -- | -- | -- | -- | -- | -- |
| WordPiece | Customer Service Dialogue Data | 9B | 1M | P40 | BERT<sup>Google</sup> weight |
* When pre-training, we do the data preprocessing with our preprocessor including some generalization processing.
### Finetune
We feed the train data into the model, and just train 2 epoches with a proper learning rate.
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
| window size | dynamic window | sub-sampling | low-frequency word | iter | negative sampling<sup>for SGNS</sup> |
| -- | -- | -- | -- | -- | -- |
| 5 | Yes | 1e-5 | 10 | 10 | 5 |
* When pre-training, we use our tools for preprocessing and word segmentaion.
### Evaluation
We evaluate our word embedding on the analogy task and the QQ similarity task. In the analogy task, we do evaluation on CA8 dataset(nature domain) and our test data to compare the performance on the e-commercial data with on data in other domain.In the chart below, CWV is abbreviation of Chinese-Word-Vectors, a open source word embedding. 
#### Analogy Task
| Model | CA8(nature) |
| -- | -- |
| CWV | 27.6 |
| Our | 13.4 |
| Model | test |
| -- | -- |
| CMV | 12.8 |
| Our | 34.0 |
#### QQ Similarity
| Model | test |
| -- | -- |
| CMV | |
| Our | |
