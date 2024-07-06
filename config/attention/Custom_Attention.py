import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow logging

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from config.arg_parser import parameter_parser

args = parameter_parser()

class CustomAttention(Layer):
    def __init__(self, units, dropout_rate=args.dropout, **kwargs):
        self.units = units
        self.dropout_rate = dropout_rate
        super(CustomAttention, self).__init__(**kwargs)

    @property
    def get_num_units(self):
        return self.units

    @property
    def get_dropout_rate(self):
        return self.dropout_rate

    def build(self, input_shape):
        # Ensure input_shape is fully specified.
        # assert len(input_shape) == 3, "Input should have shape (batch_size, sequence_length, feature_dim)"
        self.feature_dim = input_shape[-1]

        # Define trainable weights for attention mechanism
        self.W_q = self.add_weight(name='W_q', shape=(self.feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)
        self.W_k = self.add_weight(name='W_k', shape=(self.feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)
        self.W_v = self.add_weight(name='W_v', shape=(self.feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)

        # Add dropout layer
        self.dropout = Dropout(self.dropout_rate)

        super(CustomAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        q = tf.matmul(inputs, tf.expand_dims(self.W_q, axis=0))
        k = tf.matmul(inputs, tf.expand_dims(self.W_k, axis=0))
        v = tf.matmul(inputs, tf.expand_dims(self.W_v, axis=0))

        # Add dropout layer
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores)
        attention_output = tf.matmul(attention_weights, v)

        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(CustomAttention, self).get_config()
        config.update({
            "units": self.units,
        })
        return config

