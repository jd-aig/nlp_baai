import random
import numpy as np

from .beam_search import Beam
from utils_tf import  SOS_ID, UNK_ID, EOS_ID
import tensorflow as tf
import math


class BaseRNNDecoder(tf.keras.Model):
    def __init__(self):
        super(BaseRNNDecoder, self).__init__()

    def init_token(self, batch_size, SOS_ID=SOS_ID):
        """Get Variable of <SOS> Index (batch_size)"""
        x = tf.convert_to_tensor([SOS_ID] * batch_size)
        return x

    def init_h(self, batch_size=None,  hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        return tf.zeros((self.num_layers,batch_size,self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
        """
        if inputs is not None:
            batch_size = inputs.shape[0]
            return batch_size

        else:

            batch_size = h.shape[1]
            return batch_size

    def beam_decode(self,
                    init_h=None,
                    encoder_outputs=None, input_valid_length=None,
                    decode=False):
        """
        Args:
            encoder_outputs (Variable, FloatTensor): [batch_size, source_length, hidden_size]
            input_valid_length (Variable, LongTensor): [batch_size] (optional)
            init_h (variable, FloatTensor): [batch_size, hidden_size] (optional)
        Return:
            out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(h=init_h)

        # [batch_size x beam_size]
        x = self.init_token(batch_size * self.beam_size, SOS_ID)

        # [num_layers, batch_size x beam_size, hidden_size]
        h = tf.tile(self.init_h(batch_size, hidden=init_h), [1, self.beam_size, 1])

        # batch_position [batch_size]
        #   [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]
        #   Points where batch starts in [batch_size x beam_size] tensors
        #   Ex. position_idx[5]: when 5-th batch starts
        batch_position = tf.range(0, batch_size, dtype=tf.int32) * self.beam_size

        # Initialize scores of sequence
        # [batch_size x beam_size]
        # Ex. batch_size: 5, beam_size: 3
        # [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        indice = tf.reshape(batch_position, [-1, 1])
        shape = tf.constant([batch_size * self.beam_size])
        updates = tf.constant([1] * batch_size)
        score = tf.cast((tf.scatter_nd(indice, updates, shape) - 1), tf.float32) * float(9999999999)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            self.hidden_size,
            self.vocab_size,
            self.beam_size,
            self.max_unroll,
            batch_position)

        for i in range(self.max_unroll):
            # x: [batch_size x beam_size]; (token index)
            # =>
            # out: [batch_size x beam_size, vocab_size]
            # h: [num_layers, batch_size x beam_size, hidden_size]
            out, h = self.forward_step(x, h,
                                       encoder_outputs=encoder_outputs,
                                       input_valid_length=input_valid_length)
            # log_prob: [batch_size x beam_size, vocab_size]
            log_prob = tf.nn.softmax(out, axis=1)

            # [batch_size x beam_size]
            # => [batch_size x beam_size, vocab_size]
            score = tf.reshape(score, [-1, 1]) + log_prob

            # Select `beam size` transitions out of `vocab size` combinations

            # [batch_size x beam_size, vocab_size]
            # => [batch_size, beam_size x vocab_size]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_size]
            # top_k_idx: [batch_size, beam_size]
            #       each element of top_k_idx [0 ~ beam x vocab)

            score, top_k_idx = tf.math.top_k(tf.reshape(score, [batch_size, -1]), self.beam_size)

            # Get token ids with remainder after dividing by top_k_idx
            # Each element is among [0, vocab_size)
            # Ex. Index of token 3 in beam 4
            # (4 * vocab size) + 3 => 3
            # x: [batch_size x beam_size]
            x = tf.reshape((top_k_idx % self.vocab_size), [-1])

            # top-k-pointer [batch_size x beam_size]
            #       Points top-k beam that scored best at current step
            #       Later used as back-pointer at backtracking
            #       Each element is beam index: 0 ~ beam_size
            #                     + position index: 0 ~ beam_size x (batch_size-1)
            beam_idx = tf.cast((top_k_idx / self.vocab_size),tf.int32)  # [batch_size, beam_size]
            top_k_pointer = tf.reshape((beam_idx + tf.expand_dims(batch_position, 1)), [-1])

            # Select next h (size doesn't change)
            # [num_layers, batch_size * beam_size, hidden_size]
            h = tf.gather(h, top_k_pointer, axis=1)

            # Update sequence scores at beam
            beam.update(score, top_k_pointer, x)  # , h)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = tf.reshape(tf.math.equal(x, EOS_ID), [batch_size, self.beam_size])
            if tf.where(eos_idx).shape[0] > 0:
                score = tf.where(eos_idx,-float('inf'),score)

        # prediction ([batch, k, max_unroll])
        #     A list of Tensors containing predicted sequence
        # final_score [batch, k]
        #     A list containing the final scores for all top-k sequences
        # length [batch, k]
        #     A list specifying the length of each sequence in the top-k candidates
        # prediction, final_score, length = beam.backtrack()
        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length        

    def decode(self, out):
        """
        Args:
            out: unnormalized word distribution [batch_size, vocab_size]
        Return:
            x: word_index [batch_size]
        """

        # Sample next word from multinomial word distribution
        if self.sample:
            x = tf.random.categorical(tf.nn.softmax(out / self.temperature))
            x = tf.reshape(x, [-1])

        # Greedy sampling
        else:
            x = tf.math.reduce_max(out, 1)
        return x

    def embed(self, x):
        """word index: [batch_size] => word vectors: [batch_size, hidden_size]"""

        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(tf.convert_to_tensor([UNK_ID] * x.shape[0]))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed


class DecoderRNN(BaseRNNDecoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnncell=tf.keras.layers.GRUCell, num_layers=1,
                 dropout=0.0, word_drop=0.0,
                 max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.training = True
        self.beam_size = beam_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnncell = rnncell(hidden_size)
        self.out = tf.keras.layers.Dense(vocab_size)

    def forward_step(self, x, h,
                     encoder_outputs=None,
                     input_valid_length=None):
        # x: [batch_size] => [batch_size, hidden_size]
        x = self.embed(x)

        # last_h: [batch_size, hidden_size] (h from Top RNN layer)
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        last_h, h = self.rnncell(x, h)
        out = self.out(last_h)
        return out, h

    def call(self, inputs, init_h=None, encoder_outputs=None, input_valid_length=None,
                decode=False):
        """
                Train (decode=False)
                    Args:
                        inputs (Variable, LongTensor): [batch_size, seq_len]
                        init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
                    Return:
                        out   : [batch_size, seq_len, vocab_size]
                Test (decode=True)
                    Args:
                        inputs: None
                        init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
                    Return:
                        out   : [batch_size, seq_len]
                """
        batch_size = self.batch_size(inputs, init_h)

        # x: [batch_size]
        x = self.init_token(batch_size, SOS_ID)

        # h: [num_layers, batch_size, hidden_size]
        h = self.init_h(batch_size, hidden=init_h)
        
        if not decode:
            out_list = []
            seq_len = inputs.shape[1]
            for i in range(seq_len):
                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                out_list.append(out)
                x = inputs[:, i]

            # [batch_size, max_target_len, vocab_size]
            out_list = tf.convert_to_tensor(out_list)
            out = tf.transpose(out_list, [1,0,2])
            
            return out
        else:
            x_list = []
            for i in range(self.max_unroll):

                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                x = self.decode(out)
                x_list.append(x)

            # [batch_size, max_target_len]
            return tf.stack(x_list, axis=1)


if __name__ == '__main__':
    decoder = DecoderRNN(50, 100, 128)
    x = np.random.random([32,10])
    example_input_batch = tf.convert_to_tensor(x)
    sample_output = decoder(example_input_batch)
    print(sample_output.shape)
















