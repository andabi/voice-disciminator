# -*- coding: utf-8 -*-
# !/usr/bin/env python

""" A set of models """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.train.tower import get_current_tower_context

from hparam import hparam as hp
from modules.cbhg import conv1d, normalize, highwaynet, gru
from modules.wavenet import WaveNet


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

    def discriminate(self, wav, melspec, is_training=False, threshold=0.9, name='discriminator'):
        """
        :param melspec: shape=(n, t, n_mels)
        :param is_training
        :param threshold
        :param name
        :return: prediction. shape=(n, 1)
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            out = conv1d(melspec, self.hidden_units, 3, scope='conv1d_1')  # (n, t, h)
            out = normalize(out, type=self.norm_type, is_training=is_training, activation_fn=tf.nn.leaky_relu,
                            scope='norm1')
            out = conv1d(out, self.hidden_units, 3, scope='conv1d_2')  # (n, t, h)

            for i in range(self.num_highway):
                out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

            out = gru(out, self.hidden_units, False)  # (n, t, h)

            # take the last output
            out = out[..., -1, :]  # (n, h)

            # discrimination
            out = tf.layers.dense(out, self.hidden_units // 2, activation=tf.nn.leaky_relu)  # (n, h/2)
            out = normalize(out, type=self.norm_type, is_training=is_training, scope='norm2')
            out = tf.layers.dense(out, self.hidden_units // 4, activation=tf.nn.leaky_relu)  # (n, h/4)
            out = normalize(out, type=self.norm_type, is_training=is_training, scope='norm3')
            # prob = tf.layers.dense(out, 1, name='prob', activation=tf.nn.sigmoid)  # (n, 1)
            logits = tf.layers.dense(out, 2, name='prob')  # (n, 2)
            prob = tf.nn.softmax(logits)  # (n, 2)
            pred = tf.greater(prob, threshold)  # (n, 2)
        return logits, prob, pred

    def loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        # tar_labels, ntar_labels = self.labels[:, 1], self.labels[:, 0]
        # loss = -(tar_labels * tf.log(self.prob) + ntar_labels * tf.log(1. - self.prob))  # cross entropy
        loss = tf.reduce_mean(loss, name='loss')
        return loss

    def _get_inputs(self):
        length_melspec = hp.signal.length // hp.signal.hop_length + 1
        return [InputDesc(tf.float32, (None, hp.signal.length), 'wav'),
                InputDesc(tf.float32, (None, length_melspec, hp.signal.n_mels), 'melspec'),
                InputDesc(tf.float32, (None, 2), 'labels')]

    def _build_graph(self, inputs):
        self.wav, self.melspec, self.labels = inputs
        is_training = get_current_tower_context().is_training
        self.logits, self.prob, self.pred = self.discriminate(self.wav, self.melspec, is_training,
                                                              threshold=hp.disc.threshold)  # (n, 1), (n, 1), (n, 1)
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('train/loss', self.cost)
        # tf.summary.scalar('train/accuracy', self.accuracy())

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)


class WaveNetClassificationModel(ModelDesc):
    def __init__(self, dilations, residual_channels, dilation_channels, quantization_channels, skip_channels,
                 dropout=0):
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.skip_channels = skip_channels
        self.dropout = dropout

    def discriminate(self, wav, melspec, is_training=False, threshold=0.9, name='discriminator'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # prenet for dropout
            out = tf.expand_dims(wav, axis=-1)  # (n, t, 1)
            out = tf.layers.dense(out, self.quantization_channels // 2, name='dense1')  # (n, t, h/2)
            out = tf.layers.dropout(out, rate=self.dropout, training=is_training, name='dropout1')
            out = tf.layers.dense(out, self.quantization_channels, name='dense2')  # (n, t, h)
            out = tf.layers.dropout(out, rate=self.dropout, training=is_training, name='dropout2')

            wavenet = WaveNet(
                dilations=self.dilations,
                residual_channels=self.residual_channels,
                dilation_channels=self.dilation_channels,
                quantization_channels=self.quantization_channels,
                skip_channels=self.skip_channels,
                use_biases=True,
                is_training=is_training)
            out = wavenet(out)

            # take the last output
            out = out[..., -1, :]  # (n, h)

            # discrimination
            out = tf.layers.dense(out, self.quantization_channels // 2, activation=tf.nn.leaky_relu)  # (n, h/2)
            logits = tf.layers.dense(out, 2, name='prob')  # (n, 2)
            prob = tf.nn.softmax(logits)  # (n, 2)
            pred = tf.greater(prob, threshold)  # (n, 2)
        return logits, prob, pred

    def loss(self, alpha=10.):
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        tar_labels, ntar_labels = self.labels[:, 1], self.labels[:, 0]
        tar_prob, ntar_prob = self.prob[:, 1], self.prob[:, 0]
        loss = -(tar_labels * tf.log(tar_prob) + alpha * ntar_labels * tf.log(ntar_prob))  # cross entropy
        loss = tf.reduce_mean(loss, name='loss')
        return loss

    def _get_inputs(self):
        length_melspec = hp.signal.length // hp.signal.hop_length + 1
        return [InputDesc(tf.float32, (None, hp.signal.length), 'wav'),
                InputDesc(tf.float32, (None, length_melspec, hp.signal.n_mels), 'melspec'),
                InputDesc(tf.float32, (None, 2), 'labels')]

    def _build_graph(self, inputs):
        self.wav, _, self.labels = inputs
        is_training = get_current_tower_context().is_training
        self.logits, self.prob, self.pred = self.discriminate(self.wav, is_training,
                                                              threshold=hp.disc.threshold)
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('train/loss', self.cost)
        # tf.summary.scalar('train/accuracy', self.accuracy())

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)


class DefaultModel(BinaryClassificationModel):
    def __init__(self):
        model_params = {
            'num_highway': 4,
            'hidden_units': 128,
            'norm_type': 'ins'
        }
        super().__init__(**model_params)


class WaveNet3072ClsModel(WaveNetClassificationModel):
    def __init__(self):
        model_params = {
            'dilations':
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            'residual_channels': 16,
            'dilation_channels': 16,
            'skip_channels': 32,
            'quantization_channels': 32
        }
        super().__init__(**model_params)


class WaveNet1024ClsModel(WaveNetClassificationModel):
    def __init__(self):
        model_params = {
            'dilations':
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            'residual_channels': 16,
            'dilation_channels': 16,
            'skip_channels': 32,
            'quantization_channels': 32
        }
        super().__init__(**model_params)


class WaveNetDropout1024ClsModel(WaveNetClassificationModel):
    def __init__(self):
        model_params = {
            'dilations':
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            'residual_channels': 16,
            'dilation_channels': 16,
            'skip_channels': 32,
            'quantization_channels': 32,
            'dropout': 0.5
        }
        super().__init__(**model_params)