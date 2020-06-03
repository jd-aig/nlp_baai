from keras.applications.mobilenet import preprocess_input
import numpy as np
import copy
import keras as Keras
from tensorflow.keras.layers import GRU, Embedding
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras


class ImageEncoder(tf.keras.Model):
    def __init__(self, feature_size):
        super(ImageEncoder, self).__init__()
        self.feature_size = feature_size
        self.model = Keras.applications.mobilenet.MobileNet(input_shape = (224,224,3),include_top=False, weights='imagenet', pooling='avg')
        self.model.trainable = False
        self.trans = tf.keras.layers.Dense(self.feature_size)

    def call(self, inputs, training=None, mask=None):
        inputs = preprocess_input(inputs)
        inputs = self.model.predict(inputs, steps=1)
        inputs = self.trans(tf.convert_to_tensor(inputs,tf.float32))
        return inputs


class BaseRNNEncoder(tf.keras.Model):
    def __init__(self):
        super(BaseRNNEncoder, self).__init__()

    def init_h(self, batch_size=None, hidden=None):
        if hidden is not None:
            return hidden
        else:
            return [tf.zeros((batch_size,self.hidden_size))]*self.num_directions


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size,
               hidden_size, rnn = GRU, dropout=0.0, bias = True, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embedding_size,mask_zero=True)
        self.rnn = rnn(units=self.hidden_size,
                       return_sequences=True,
                       return_state=True,
                       dropout=dropout,
                       use_bias=bias
                       )
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)
            self.num_directions = 2
        else:
            self.num_directions=1

    def call(self, inputs, input_length, hidden=None):
        batch_size, seq_len = inputs.shape
        max_indice = tf.argmax(input_length)
        seq_len = int(input_length[max_indice])
        inputs = tf.slice(inputs, [0, 0], [-1, seq_len])
        hidden = self.init_h(batch_size, hidden=hidden)
        embedded = self.embedding(inputs)
        mask = tf.sequence_mask(input_length, seq_len)

        if self.bidirectional:
            output, s1, s2 = self.rnn(inputs=embedded, mask=mask, initial_state=hidden)
            state = [s1, s2]
        else:
            output, s = self.rnn(inputs=embedded, mask=mask, initial_state=hidden)
            state = tf.expand_dims(s,1)
        return output[0], state


class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=GRU, dropout=0.0,
                 bias=True, batch_first=True,num_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = context_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_first = batch_first
        self.rnn = rnn(units=self.hidden_size,
                       return_sequences=True,
                       return_state=True,
                       dropout=dropout,
                       use_bias=bias
                       )

    def call(self, encoder_hidden, conversation_length, hidden=None):
        """
                Args:
                    encoder_hidden (Variable, FloatTensor): [batch_size, max_len(10), (1024)num_layers * direction * hidden_size]
                    conversation_length (Variable, LongTensor): [batch_size]
                Return:
                    outputs (Variable): [batch_size, max_seq_len, hidden_size]
                        - list of all hidden states
                    hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                        - last hidden state
                        - (h, c) or h
        """
        batch_size, seq_len, _ = encoder_hidden.shape
        #取最大的seqlen
        seq_len = int(conversation_length[tf.argmax(conversation_length)])
        encoder_hidden = tf.slice(encoder_hidden, [0,0,0],[-1,seq_len,-1])*10
        mask = tf.sequence_mask(conversation_length,seq_len)
        hidden = self.init_h(batch_size, hidden)
        output, s = self.rnn(inputs=encoder_hidden, mask=mask, initial_state=hidden)
        state = tf.expand_dims(s, 1)
        return output, state

    def step(self, encoder_hidden, hidden):
        batch_size = encoder_hidden.shape[0]
        encoder_hidden = tf.expand_dims(encoder_hidden,1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)
        output, s = self.rnn(inputs=encoder_hidden,  initial_state=hidden)
        state = tf.expand_dims(s, 1)
        return output, state


if __name__ == '__main__':
    img_encoder = ImageEncoder(1024)
    t = tf.convert_to_tensor(np.random.random([224,32*32*3]),tf.float32)
    t = tf.keras.layers.Dense(32*32*3)(t)
    t = tf.reshape(t,[224,32,32,3])
    out = img_encoder(t)
    print(out.shape)

