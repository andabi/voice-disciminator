# -*- coding: utf-8 -*-
#!/usr/bin/env python

""" A set of models """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.train.tower import get_current_tower_context

from hparam import hparam as hp
from modules import conv1d_banks, conv1d, normalize, highwaynet, gru


class BinaryClassificationModel(ModelDesc):
    """
    n = batch size
    t = timestep size
    h = hidden size
    """

    def __init__(self, hidden_units, num_highway, norm_type):
        self.hidden_units = hidden_units
        self.num_highway = num_highway
        self.norm_type = norm_type

    def discriminate(self, x, is_training=False, t=1., threshold=0.85, name='discriminator'):
        """
        :param x: shape=(n, t, n_mels)
        :param is_training
        :param t: temperature
        :param threshold
        :param name
        :return: prediction. shape=(n, 1)
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            out = conv1d(x, self.hidden_units, 3, scope='conv1d_1')  # (n, t, h)
            out = normalize(out, type=self.norm_type, is_training=is_training, activation_fn=tf.nn.relu)
            out = conv1d(out, self.hidden_units, 3, scope='conv1d_2')  # (n, t, h)

            for i in range(self.num_highway):
                out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

            out = gru(out, self.hidden_units, False)  # (n, t, h)

            # take the last output
            out = out[..., -1, :]  # (n, h)

            # discrimination
            logits = tf.layers.dense(out, 2, name='logits')  # (n, 2)
            prob = tf.nn.softmax(logits / t)  # (n, 2)
            pred = tf.greater(prob[:, 1], threshold)  # (n,)

        return logits, prob, pred

    def loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        loss = tf.reduce_mean(loss, name='loss')
        return loss

    def _get_inputs(self):
        length_melspec = hp.signal.length // hp.signal.hop_length + 1
        return [InputDesc(tf.float32, (None, hp.signal.length), 'wav'),
                InputDesc(tf.float32, (None, length_melspec, hp.signal.n_mels), 'melspec'),
                InputDesc(tf.float32, (None, 2), 'labels')]

    def _build_graph(self, inputs):
        _, self.x, self.labels = inputs
        is_training = get_current_tower_context().is_training
        self.logits, self.prob, self.pred = self.discriminate(self.x, is_training)  # (n, 2), (n,)
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('train/loss', self.cost)
        # tf.summary.scalar('train/accuracy', self.accuracy())

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)


class CBHGModel(BinaryClassificationModel):
    """
    n = batch size
    t = timestep size
    h = hidden size
    """

    def __init__(self, num_banks, hidden_units, num_highway, norm_type):
        super().__init__(hidden_units, num_highway, norm_type)
        self.num_banks = num_banks

    def discriminate(self, x, is_training=False, t=1., threshold=0.85, name='discriminator'):
        """
        :param x: shape=(n, t, n_mels)
        :param is_training
        :param t: temperature
        :param threshold
        :param name
        :return: prediction. shape=(n, 1)
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # frame-level
            x = tf.layers.dense(x, units=self.hidden_units, activation=tf.nn.relu)  # (n, t, h)

            out = conv1d_banks(x, K=self.num_banks, num_units=self.hidden_units, norm_type=self.norm_type,
                               is_training=is_training)  # (n, t, k * h)

            out = tf.layers.max_pooling1d(out, 2, 1, padding='same')  # (n, t, k * h)

            out = conv1d(out, self.hidden_units, 3, scope='conv1d_1')  # (n, t, h)
            out = normalize(out, type=self.norm_type, is_training=is_training, activation_fn=tf.nn.relu)
            out = conv1d(out, self.hidden_units, 3, scope='conv1d_2')  # (n, t, h)
            out += x  # (n, t, h) # residual connections

            for i in range(self.num_highway):
                out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

            out = gru(out, self.hidden_units, False)  # (n, t, h)

            # take the last output
            out = out[..., -1, :]  # (n, h)

            # discrimination
            logits = tf.layers.dense(out, 2, name='logits')  # (n, 2)
            prob = tf.nn.softmax(logits / t)  # (n, 2)
            pred = tf.greater(prob[:, 1], threshold)  # (n,)

        return logits, prob, pred


# default_model = CBHGModel(num_banks=8, hidden_units=128, num_highway=2, norm_type='ins')
default_model = BinaryClassificationModel(hidden_units=64, num_highway=2, norm_type='ins')