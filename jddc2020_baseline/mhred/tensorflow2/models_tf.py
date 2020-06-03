
from utils_tf import Vocab
from utils_tf import pad,  bag_of_words_loss, to_bow, EOS_ID
import layers_tf
import tensorflow as tf
from configs_tf import get_config
import keras_applications
import numpy as np

VariationalModels = ['VHRED', 'VHCR']


class HRED(tf.keras.Model):
    def __init__(self, config):
        super(HRED, self).__init__()

        self.config = config
        self.encoder = layers_tf.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         dropout=config.dropout,
                                         bidirectional=config.bidirectional)
        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)

        self.image_encoder = layers_tf.ImageEncoder(context_input_size)

        self.context_encoder = layers_tf.ContextRNN(context_input_size*2,
                                                 config.context_size,
                                                 num_layers=config.num_layers,
                                                 dropout=config.dropout)

        self.decoder = layers_tf.DecoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                         num_layers=config.num_layers,
                                         dropout=config.dropout,
                                         word_drop=config.word_drop,
                                         max_unroll=config.max_unroll,
                                         sample=config.sample,
                                         temperature=config.temperature,
                                         beam_size=config.beam_size)
        self.feedforward = tf.keras.layers.Dense(self.config.decoder_hidden_size,self.config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def call(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences,
                input_images, input_images_length,
                decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.shape[0]
        max_len = int(tf.reduce_max(input_conversation_length))

        batch_size = input_conversation_length.shape[0]
        _, encoder_hidden = self.encoder(input_sentences,input_sentence_length)

        input_images = tf.reshape(input_images, [batch_size, -1, 3, 224, 224])
        input_images_length = tf.reshape(input_images_length, [batch_size, -1])
        input_images = tf.slice(input_images, [0]*5, [-1, max_len]+[-1]*3)
        input_images = tf.reshape(input_images, [-1, 224, 224, 3])
        input_images_length = tf.slice(input_images_length, [0, 0], [-1, max_len])

        img_encoder_outputs = self.image_encoder(input_images)
        img_encoder_outputs = tf.reshape(img_encoder_outputs, [batch_size, max_len, -1])
        input_images_length = tf.expand_dims(input_images_length, axis=-1)
        img_encoder_outputs = img_encoder_outputs * input_images_length

        if self.encoder.bidirectional:
            encoder_hidden = tf.stack(encoder_hidden, axis=1)
        encoder_hidden = tf.reshape(encoder_hidden, [num_sentences,-1])
        start = tf.cumsum(tf.concat((tf.convert_to_tensor([0], dtype=tf.float32), input_conversation_length[:-1]), 0))

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden = tf.stack([pad(encoder_hidden[int(s):int(s+l)], max_len)
                                      for s, l in zip(start, input_conversation_length)], 0)
        comb_encoder_hidden = tf.concat([encoder_hidden, img_encoder_outputs], 2)
        context_outputs, _ = self.context_encoder(comb_encoder_hidden, input_conversation_length)
        context_outputs = tf.concat([context_outputs[i, :int(l), :]
                                     for i, l in enumerate(input_conversation_length)], 0)

        context_outputs = self.feedforward(context_outputs)
        decoder_init = tf.reshape(context_outputs,[self.decoder.num_layers, -1, self.decoder.hidden_size])
        
        if not decode:
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs
        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction

    def generate(self, context, sentence_length, n_context, input_images, input_images_length):
        batch_size = context.shape[0]
        samples = []

        max_len = n_context

        # Run for context
        context_hidden = None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context,sentence_length)

            encoder_hidden = tf.reshape(tf.transpose(encoder_hidden,[1, 0,2]), [batch_size, -1])

            input_image = tf.expand_dims(input_images[i], 0)
            img_encoder_outputs = self.image_encoder(input_image)
            img_encoder_outputs = tf.reshape(img_encoder_outputs, [1, -1])
            input_images_length = tf.expand_dims(input_images_length, axis=-1)
            img_encoder_outputs = img_encoder_outputs * input_images_length
            comb_encoder_hidden = tf.concat([encoder_hidden, img_encoder_outputs], 1)
            context_outputs, context_hidden = self.context_encoder.step(comb_encoder_hidden,
                                                                        context_hidden)
        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = tf.squeeze(context_outputs, 1)
            decoder_init = tf.reshape(context_outputs, [self.decoder.num_layers, -1, self.decoder.hidden_size])
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            # prediction: [batch_size, seq_len]
            prediction = prediction[:, 0, :] #选最好的
            # length: [batch_size]
            length = [l[0] for l in length]
            length = tf.convert_to_tensor(length)
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction, length)
            encoder_hidden = tf.reshape(tf.transpose(encoder_hidden, [1, 0, 2]), [batch_size, -1])
            img_encoder_pad = tf.zeros(encoder_hidden.shape)
            comb_encoder_hidden = tf.concat([encoder_hidden, img_encoder_pad], 1)

            context_outputs, context_hidden = self.context_encoder.step(comb_encoder_hidden,
                                                                        context_hidden)

        samples = tf.stack(samples, 1)
        return samples


if __name__ == '__main__':
    config = get_config(mode='train')
    print('Loading Vocabulary...')

    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size
    model = HRED(config)

    sent = tf.zeros([151,50])
    tarsent = tf.zeros([151,50])
    sent_len = tf.convert_to_tensor([4]*151)
    conv_len = tf.convert_to_tensor([2,2,2,2,3,3,3,3]*4)
    img = tf.convert_to_tensor(np.random.random([320,3,244,244]))
    img_len = tf.convert_to_tensor([0,1]*160)

    logit = model(sent, sent_len, conv_len, tarsent, img, img_len)
    print(logit.shape)
    print(tarsent.shape)
