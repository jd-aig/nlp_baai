import tensorflow as tf
import pickle as plk
import numpy as np
from utils_tf import vocab
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pdb


class Dataloader():
    def __init__(self, sent, sent_len, conv_len, img, img_len, batch_size, vocab, mode):
        self.sent = sent
        self.sent_len = sent_len
        self.conv_len = conv_len
        self.img = img
        self.img_len = img_len
        self.batch_size = batch_size
        self.vocab = vocab
        self.mode = mode

    def img_gen(self):
        i = 0
        images = self.img
        while i<len(images):
            re_list = []
            bdata = images[i:i+self.batch_size]
            bdata = sum(bdata,[])
            for loc in bdata:
                if loc == 'NULL':
                    re_list.append(tf.zeros([224,224,3]))
                else:
                    img = tf.zeros([224,224,3])
                    try:
                        locc = loc.split('_')[-1]
                        if self.mode == 'test':
                            img_tmp = load_img('./data/online_test_data/images_test/' + locc)
                        else:
                            img_tmp = load_img('./data/images/' + self.mode + '/' + locc)

                        img = img_to_array(img_tmp)
                        img = tf.image.resize(img,[224,224])/255
                    except:
                        None
                    finally:
                        re_list.append(img)

            re_list = tf.convert_to_tensor(re_list)

            yield re_list
            i += self.batch_size

    def sent2id(self, sentence):
        """list of tokens => list of id (Single sentence)"""
        sent_list = [self.vocab.sent2id(sent) for sent in sentence]

        return sent_list

    def sent_gen(self):
        i = 0

        sentences = self.sent
        while(i<len(sentences)):
            bdata = sentences[i:i+self.batch_size]
            bdata = [con[:-1] for con in bdata]
            bdata = sum(bdata,[])
            bdata = self.sent2id(bdata)
        
            yield bdata
            i += self.batch_size
            
    def target_gen(self):
        i = 0
        sentences = self.sent
        while(i<len(sentences)):

            bdata = sentences[i:i+self.batch_size]
            bdata = [con[1:] for con in bdata]
            bdata = sum(bdata,[])
            bdata = self.sent2id(bdata)
            yield bdata
            i += self.batch_size

    def sent_len_gen(self):
        sent_len = self.sent_len
        i = 0
        while(i<len(sent_len)):
            bdata = sent_len[i:i+self.batch_size]
            bdata = [senlen[:-1] for senlen in bdata]
            bdata = sum(bdata,[])
            i += self.batch_size
            yield bdata

    def target_len_gen(self):
        sent_len = self.sent_len
        i = 0
        while(i<len(sent_len)):
            bdata = sent_len[i:i+self.batch_size]
            bdata = [senlen[1:] for senlen in bdata]
            bdata = sum(bdata,[])
            i += self.batch_size
            yield bdata

    def cov_len_gen(self):
        conv_len = self.conv_len
        i = 0
        while(i<len(conv_len)):
            bdata = conv_len[i:i+self.batch_size]
            bdata = np.array(bdata) - 1
            i += self.batch_size
            yield bdata

    def img_len_gen(self):
        img_len = self.img_len
        i = 0
        while(i<len(img_len)):
            bdata = img_len[i:i+self.batch_size]
            bdata = sum(bdata,[])
            i += self.batch_size
            yield bdata

    def get_data_loader(self):

        sent = tf.data.Dataset.from_generator(self.sent_gen, tf.float32)
        target_sent = tf.data.Dataset.from_generator(self.target_gen, tf.float32)

        sent_len = tf.data.Dataset.from_generator(self.sent_len_gen, tf.float32)
        target_len = tf.data.Dataset.from_generator(self.target_len_gen, tf.float32)

        conv_len = tf.data.Dataset.from_generator(self.cov_len_gen, tf.float32)
        img = tf.data.Dataset.from_generator(self.img_gen, tf.float32)

        img_len = tf.data.Dataset.from_generator(self.img_len_gen, tf.float32)
        data = tf.data.Dataset.zip((sent, target_sent, conv_len, sent_len, target_len, img, img_len)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data



