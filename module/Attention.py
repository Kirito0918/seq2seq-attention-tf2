import tensorflow as tf
import tensorflow.keras as keras

class Attention(keras.layers.Layer):

    def __init__(self, decoder_output_size,
                 encoder_output_size,
                 attention_type,
                 attention_size):
        super(Attention, self).__init__()
        assert attention_type in ['dot', 'general', 'concat', 'perceptron']  # 点乘、权重、拼接权重、感知机

        self.linear_q = keras.layers.Dense(attention_size, input_shape=(None, None, decoder_output_size),
                                           activation='relu')
        self.linear_k = keras.layers.Dense(attention_size, input_shape=(None, None, encoder_output_size),
                                           activation='relu')
        self.linear_v = keras.layers.Dense(attention_size, input_shape=(None, None, encoder_output_size),
                                           activation='relu')

        if attention_type == 'general':
            self.general_w = keras.layers.Dense(attention_size, input_shape=(None, None, attention_size), use_bias=False)
        elif attention_type == 'concat':
            self.concat_w = keras.layers.Dense(1, input_shape=(None, None, attention_size*2), use_bias=False)
        elif attention_type == 'perceptron':
            self.perceptron_w = keras.layers.Dense(attention_size, input_shape=(None, None, attention_size), use_bias=False)
            self.perceptron_u = keras.layers.Dense(attention_size, input_shape=(None, None, attention_size), use_bias=False)
            self.perceptron_v = keras.layers.Dense(1, input_shape=(None, None, attention_size), use_bias=False)

        self.attention_type = attention_type
        self.attention_size = attention_size


    def __call__(self, encoder_outputs,  # [batch_size, encoder_len, output_size]
                 decoder_outputs):  # [batch_size, decoder_len, output_size]

        querys = self.linear_q(decoder_outputs)  # [batch_size, decoder_len, attention_size]
        keys = self.linear_k(encoder_outputs)  # [batch_size, encoder_len, attention_size]
        values = self.linear_v(encoder_outputs)  # [batch_size, encoder_len, attention_size]

        if self.attention_type == 'dot':

            weights = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # [batch_size, decoder_len, encoder_len]
            weights = weights / self.attention_size ** 0.5
            weights = tf.math.softmax(weights, -1)
            attention = tf.matmul(weights, values)  # [batch_size, decoder_len, attention_size]

        elif self.attention_type == 'general':

            keys = self.general_w(keys)  # [batch_size, encoder_len, attention_size]
            weights = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # [batch_size, decoder_len, encoder_len]
            weights = weights / self.attention_size ** 0.5
            weights = tf.math.softmax(weights, -1)
            attention = tf.matmul(weights, values)  # [batch_size, decoder_len, attention_size]

        elif self.attention_type == 'concat':

            decoder_len = tf.shape(querys)[1]
            encoder_len = tf.shape(keys)[1]

            # [batch_size, decoder_len, encoder_len, attention_size]
            querys = tf.tile(tf.expand_dims(querys, 2), [1, 1, encoder_len, 1])
            keys = tf.tile(tf.expand_dims(keys, 1), [1, decoder_len, 1, 1])
            # [batch_size, decoder_len, encoder_len]
            weights = tf.reshape(self.concat_w(tf.concat([querys, keys], 3)), [-1, decoder_len, encoder_len])
            weights = weights / self.attention_size ** 0.5
            weights = tf.math.softmax(weights, -1)
            attention = tf.matmul(weights, values)  # [batch_size, decoder_len, attention_size]

        else:  # 'perceptron'

            decoder_len = tf.shape(querys)[1]
            encoder_len = tf.shape(keys)[1]

            # [batch_size, decoder_len, encoder_len, attention_size]
            querys = tf.tile(tf.expand_dims(querys, 2), [1, 1, encoder_len, 1])
            keys = tf.tile(tf.expand_dims(keys, 1), [1, decoder_len, 1, 1])

            querys = self.perceptron_w(querys)  # [batch_size, decoder_len, encoder_len, attention_size]
            keys = self.perceptron_u(keys)  # [batch_size, decoder_len, encoder_len, attention_size]
            weights = tf.reshape(self.perceptron_v(querys+keys), [-1, decoder_len, encoder_len])
            weights = weights / self.attention_size ** 0.5
            weights = tf.math.softmax(weights, -1)
            attention = tf.matmul(weights, values)  # [batch_size, decoder_len, attention_size]

        return attention
