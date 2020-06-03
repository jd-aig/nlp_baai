from itertools import cycle
import numpy as np
import tensorflow as tf
import models_tf
from data_tf import Dataloader
from layers_tf import loss_function
from utils_tf import  time_desc_decorator,   embedding_metric
import os
from tqdm import tqdm
from math import isnan
import re
import math
import pickle
import pdb


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.vocab = vocab
        self.is_train = is_train
        self.model = model
        self.checkpoint_dir = self.config.save_path
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        self.model = models_tf.HRED(self.config)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)
        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

    def load_model(self, checkpoint):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')

        epoch = os.path.basename(checkpoint)[-1]
        self.epoch_i = int(epoch)
        print(epoch, checkpoint)
        self.checkpoint.restore(checkpoint)

    def train_step(self,input_sentences, target_sentences,conversation_length,input_length,target_length,img,img_len):

        with tf.GradientTape() as tape:

            sentence_logits = self.model(
                input_sentences,
                input_length,
                conversation_length,
                target_sentences,
                img,img_len,
                decode=False)

            Variables = self.model.trainable_variables

            loss, n_words = loss_function(
                real = target_sentences,
                pred = sentence_logits,
                length = target_length,
                loss_object = self.loss_object)

        batch_loss = loss/n_words
    
        gradient = tape.gradient(loss, Variables)
        
        gradient, _ = tf.clip_by_global_norm(gradient, 1)
        self.optimizer.apply_gradients(zip(gradient, Variables))
        return batch_loss

    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        flag = 0
       
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []

            n_total_words = 0
            for batch_i, (input_sentences, target_sentences,conversation_length, input_length,
                          target_length,img,img_len) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                
                batch_loss = self.train_step(input_sentences, target_sentences, conversation_length, input_length, target_length,img,img_len)
                n_words = int(tf.reduce_sum(target_length))
                batch_loss_history.append(batch_loss)
                n_total_words += n_words

                if batch_i % self.config.print_every == 0:
                    tqdm.write(
                        f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {float(batch_loss):.3f}')

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        return epoch_loss_history

    def generate_sentence(self,input_sentences,input_sentence_length,
                                       input_conversation_length,target_sentences, img, img_len):

        generated_sentences = self.model(
            input_sentences,
            input_sentence_length,
            input_conversation_length,
            target_sentences,
            img,
            img_len,
            decode=True)

        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            f.write(f'<Epoch {self.epoch_i}>\n\n')
            tqdm.write('\n<Samples>')
            for input_sent, target_sent, output_sent in zip(input_sentences, target_sentences, generated_sentences):
                input_sent = self.vocab.decode(input_sent)
                target_sent = self.vocab.decode(target_sent)
                output_sent = '\n'.join([self.vocab.decode(sent) for sent in output_sent])
                s = '\n'.join(['Input sentence: ' + input_sent,
                               'Ground truth: ' + target_sent,
                               'Generated response: ' + output_sent + '\n'])
                f.write(s + '\n')
                print(s)
            print('')

    def evaluate(self):
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (input_sentences, target_sentences,conversation_length, input_length,
                          target_length,img,img_len) in enumerate(tqdm(self.eval_data_loader, ncols=80)):

            if batch_i == 0:
                self.generate_sentence(input_sentences,
                                       input_length,
                                       conversation_length,
                                       target_sentences,img,img_len)
                
            sentence_logits = self.model(
                input_sentences,
                input_length,
                conversation_length,
                target_sentences,
            img,img_len)

            loss, n_words = loss_function(
                real = target_sentences,
                pred = sentence_logits,
                length = target_length,
                loss_object = self.loss_object)

            batch_loss_history.append(loss)
            n_total_words += n_words

        epoch_loss = np.sum(batch_loss_history) / n_total_words

        print_str = f'Validation loss: {epoch_loss:.3f}\n'
        print(print_str)
        return epoch_loss

    def generate_for_evaluation(self):
        n_sample_step = self.config.n_sample_step
        n_sent = 0
        fo = open('result.txt', "w")
        for batch_i, (input_sentences, target_sentences, conversation_length, input_length,
                      target_length, images, images_length) in enumerate(tqdm(self.eval_data_loader, ncols=80)):

            context = input_sentences
            n_context = len(context)
            sentence_length = input_length

            generated_sentences = self.model(context,
                                             sentence_length,
                                             conversation_length,
                                             target_sentences,
                                             images,
                                             images_length,
                                             decode=True)

            sent = generated_sentences[0][0]
            sample = self.vocab.decode(sent)

            n_sent += 1
            print("gen: ", sample)
            fo.write(sample + "\n")

        print('n_sentences:', n_sent)
        print('\n')
        fo.close()
