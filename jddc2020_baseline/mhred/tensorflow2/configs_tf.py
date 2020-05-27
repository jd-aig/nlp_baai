import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import tensorflow as tf

project_dir = Path(__file__).resolve().parent
data_dir = project_dir.joinpath('./data/')
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
save_dir = project_dir.joinpath('./ckpt/')
pred_dir = project_dir.joinpath('./pred/')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = project_dir.joinpath(self.data.lower())

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')

        # Pickled Dataframes
        self.sentences_path = self.data_dir.joinpath('sentences.pkl')
        self.images_path = self.data_dir.joinpath('images.pkl')
        self.images_len_path = self.data_dir.joinpath('images_length.pkl')
        self.sentence_length_path = self.data_dir.joinpath('sentence_length.pkl')
        self.conversation_length_path = self.data_dir.joinpath('conversation_length.pkl')

        os.makedirs(pred_dir, exist_ok=True)
        self.pred_path = './pred/res_img_tf.txt'
        # Save path
        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = save_dir.joinpath(self.data, self.model+'_tf', time_now)
            self.save_path=str(self.save_path).replace('\\','/')
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Generation
    parser.add_argument('--max_unroll', type=int, default=50)
    parser.add_argument('--sample', type=str2bool, default=False,
                        help='if false, use beam search for decoding')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=5)

    # Model
    parser.add_argument('--model', type=str, default='HRED',
                        help='one of {HRED, VHRED, VHCR}')
    # Currently does not support lstm
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--tie_embedding', type=str2bool, default=True)
    parser.add_argument('--encoder_hidden_size', type=int, default=512)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--decoder_hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--context_size', type=int, default=512)
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='tanh')

    # VAE model
    parser.add_argument('--z_sent_size', type=int, default=100)
    parser.add_argument('--z_conv_size', type=int, default=100)
    parser.add_argument('--word_drop', type=float, default=0.0,
                        help='only applied to variational models')
    parser.add_argument('--kl_threshold', type=float, default=0.0)
    parser.add_argument('--kl_annealing_iter', type=int, default=25000)
    parser.add_argument('--importance_sample', type=int, default=100)
    parser.add_argument('--sentence_drop', type=float, default=0.0)

    # Generation
    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--n_sample_step', type=int, default=1)

    # BOW
    parser.add_argument('--bow', type=str2bool, default=False)

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    # Data
    parser.add_argument('--data', type=str, default='./data/')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
