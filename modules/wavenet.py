# -*- coding: utf-8 -*-
#!/usr/bin/env python

""""""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def causal_conv(value, filter_, dilation, name='causal_conv'):
    def time_to_batch(value, dilation):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def batch_to_time(value, dilation):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])

    with tf.variable_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            # for left-side padding because tf.nn.conv1d do not support left-side padding with padding='SAME'
            padded = tf.pad(transformed, [[0, 0], [filter_width - 1, 0], [0, 0]])
            conv = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)

            # Remove excess elements at the end caused by padding in time_to_batch.
            result = tf.slice(restored,
                              [0, 0, 0],
                              [-1, tf.shape(value)[1], -1])
        else:
            padded = tf.pad(value, [[0, 0], [filter_width - 1, 0], [0, 0]])
            result = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
        return result


# This WaveNet code is renovated based on https://github.com/ibab/tensorflow-wavenet
class WaveNet(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
    '''

    def __init__(self,
                 dilations,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 filter_width=2,
                 quantization_channels=2 ** 8,
                 use_biases=False,
                 condition_channels=None,
                 use_skip_connection=True,
                 is_training=True,
                 name='wavenet'):
        '''Initializes the WaveNet model.

        Args:
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
        '''
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.condition_channels = condition_channels
        self.use_skip_connection = use_skip_connection
        self.is_training = is_training
        self.name = name

    def __call__(self, input_batch, condition_batch=None):
        with tf.variable_scope(self.name):
            '''Construct the WaveNet network.'''
            outputs = []
            with tf.variable_scope('causal_layer'):
                current_layer = self._create_causal_layer(input_batch)

            # Add all defined dilation layers.
            with tf.variable_scope('dilated_stack'):
                for layer_index, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(layer_index)):
                        output, current_layer = self._create_dilation_layer(
                            current_layer, dilation, condition_batch)
                        outputs.append(output)

            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to postprocess the output.
            with tf.variable_scope('postprocessing'):
                # We skip connections from the outputs of each layer, adding them all up here.
                total = sum(outputs) if self.use_skip_connection else outputs[-1]
                transformed1 = tf.nn.relu(total)
                w1 = tf.get_variable('postprocess1', [1, self.skip_channels, self.skip_channels])
                conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
                if self.use_biases:
                    b1 = tf.get_variable('postprocess1_bias', [self.skip_channels], initializer=tf.zeros_initializer)
                    conv1 = tf.add(conv1, b1)
                transformed2 = tf.nn.relu(conv1)
                w2 = tf.get_variable('postprocess2', [1, self.skip_channels, self.quantization_channels])
                conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
                if self.use_biases:
                    b2 = tf.get_variable('postprocess2_bias', [self.quantization_channels], initializer=tf.zeros_initializer)
                    conv2 = tf.add(conv2, b2)
        return conv2

    # network
    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        weights_filter = tf.get_variable('filter', [self.filter_width, self.quantization_channels, self.residual_channels])
        layer = causal_conv(input_batch, weights_filter, 1)
        return layer

    # @staticmethod
    # def calculate_receptive_field(filter_width, dilations):
    #     receptive_field = (filter_width - 1) * sum(dilations) + 1
    #     receptive_field += filter_width - 1
    #     return receptive_field

    def _create_dilation_layer(self, input_batch, dilation, local_condition):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             local_condition: The data which each timestep is to be conditioned on. 
                Shape: [batch size, n_timesteps, channels].

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and conditioning
        are omitted due to the limits of ASCII art.

        '''

        weights_filter = tf.get_variable('filter', [self.filter_width, self.residual_channels, self.dilation_channels])
        weights_gate = tf.get_variable('gate', [self.filter_width, self.residual_channels, self.dilation_channels])

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if local_condition is not None:
            weights_cond_filter = tf.get_variable('gc_filter', [1, self.condition_channels, self.dilation_channels])
            conv_filter = conv_filter + tf.nn.conv1d(local_condition, weights_cond_filter, stride=1, padding="SAME",
                                                     name="gc_filter")
            weights_cond_gate = tf.get_variable('gc_gate', [1, self.condition_channels, self.dilation_channels])
            conv_gate = conv_gate + tf.nn.conv1d(local_condition, weights_cond_gate, stride=1, padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = tf.get_variable('filter_bias', [self.dilation_channels], initializer=tf.zeros_initializer)
            gate_bias = tf.get_variable('gate_bias', [self.dilation_channels], initializer=tf.zeros_initializer)
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = tf.get_variable('dense', [1, self.dilation_channels, self.residual_channels])
        transformed = tf.nn.conv1d(out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = tf.get_variable('skip', [1, self.dilation_channels, self.skip_channels])
        skip_output = tf.nn.conv1d(out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = tf.get_variable('dense_bias', [self.residual_channels], initializer=tf.zeros_initializer)
            skip_bias = tf.get_variable('skip_bias', [self.skip_channels], initializer=tf.zeros_initializer)
            transformed = transformed + dense_bias
            skip_output = skip_output + skip_bias
        dense_output = input_batch + transformed

        return skip_output, dense_output