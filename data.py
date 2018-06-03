# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob

from audio import read_wav, wav2melspec_db, trim_wav, crop_random_wav, fix_length, augment_volume
from hparam import hparam as hp

import tensorflow as tf
import numpy as np


class LabelledDataset():
    """
    A reference for best practices: 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/performance/datasets_performance.md#summary-of-best-practices
    """

    def __init__(self, batch_size, tar_path, ntar_path, length=4000, tar_ratio=0.5):
        self.batch_size = batch_size
        self.tar_wavfiles = glob.glob(tar_path, recursive=True)
        self.ntar_wavfiles = glob.glob(ntar_path, recursive=True)
        if len(self.tar_wavfiles) == 0:
            raise FileNotFoundError("target dataset not found.")
        print('target dataset size: {}'.format(len(self.tar_wavfiles)))
        if len(self.ntar_wavfiles) == 0:
            raise FileNotFoundError("non-target dataset not found.")
        print('non-target dataset size: {}'.format(len(self.ntar_wavfiles)))
        self.length = length
        self.tar_ratio = tar_ratio

    def __call__(self, n_prefetch=1000, n_thread=32):
        dataset = tf.data.Dataset.from_tensors((0., 0., 0))  # dummy input
        # real data generator.
        # This is workaround: use it to execute in parallel instead of using tf.data.Dataset.from_generator.
        dataset = dataset.map(
            lambda w, m, l: tf.py_func(self.get_random_wav_and_label, [], [tf.float32, tf.float32, tf.int32]),
            num_parallel_calls=n_thread)
        # reshape tensors to define the shapes manually. It will occur errors in tf.layer.dense without this.
        length_melspec = hp.signal.length // hp.signal.hop_length + 1
        dataset = dataset.map(lambda w, m, l: (tf.reshape(w, [self.length]), tf.reshape(m, [length_melspec, hp.signal.n_mels]), tf.reshape(l, [])))
        dataset = dataset.repeat().batch(self.batch_size).prefetch(n_prefetch)
        return dataset

    def get_random_wav_and_label(self):
        """

        :return: wav: raw wave. float32. shape=(t, ),
                 label: 1 if target, 0 otherwise. int32.
                 melspec: mel-spectrogram. float32. shape=(t, n_mels)
        """
        wavfiles, label = (self.tar_wavfiles, 1) if np.random.sample(1) <= self.tar_ratio else (self.ntar_wavfiles, 0)
        wavfile = np.random.choice(wavfiles, 1)[0]
        wav = read_wav(wavfile, sr=hp.signal.sr)
        wav = trim_wav(wav)
        wav = crop_random_wav(wav, self.length)
        wav = augment_volume(wav)
        wav = fix_length(wav, self.length)  # padding
        melspec = wav2melspec_db(wav, sr=hp.signal.sr, n_fft=hp.signal.n_fft, win_length=hp.signal.win_length,
                                 hop_length=hp.signal.hop_length, n_mels=hp.signal.n_mels,
                                 min_db=hp.signal.min_db, max_db=hp.signal.max_db)
        melspec = np.float32(melspec)
        label = np.int32(label)
        return wav, melspec, label


class UnLabelledDataset():

    def __init__(self, batch_size, data_path, length=4000):
        self.batch_size = batch_size
        self.wavfiles = glob.glob(data_path, recursive=True)
        if len(self.wavfiles) == 0:
            raise FileNotFoundError("dataset not found.")
        print('dataset size: {}'.format(len(self.wavfiles)))
        self.length = length

    def __call__(self, n_prefetch=1000, n_thread=32):
        length_melspec = hp.signal.length // hp.signal.hop_length + 1
        dataset = tf.data.Dataset.from_tensor_slices(self.wavfiles)
        dataset = dataset.map(
            lambda wavfile: tf.py_func(self.get_random_wav, [wavfile], [tf.float32, tf.float32, tf.string]),
            num_parallel_calls=n_thread)
        dataset = dataset.map(lambda w, m, s: (tf.reshape(w, [self.length]), tf.reshape(m, [length_melspec, hp.signal.n_mels]), tf.reshape(s, [])))
        dataset = dataset.batch(self.batch_size).prefetch(n_prefetch)
        return dataset

    def get_random_wav(self, wavfile):
        """
        :param: wavfile: a raw wave file.
        :return: wav: raw wave. float32. shape=(t, ),
                 melspec: mel-spectrogram. float32. shape=(t, n_mels),
                 wavfile: the raw wave file.
        """
        wav = read_wav(wavfile, sr=hp.signal.sr)
        wav = trim_wav(wav)
        wav = crop_random_wav(wav, self.length)
        wav = fix_length(wav, self.length)  # padding
        melspec = wav2melspec_db(wav, sr=hp.signal.sr, n_fft=hp.signal.n_fft, win_length=hp.signal.win_length,
                                 hop_length=hp.signal.hop_length, n_mels=hp.signal.n_mels,
                                 min_db=hp.signal.min_db, max_db=hp.signal.max_db)
        melspec = np.float32(melspec)
        return wav, melspec, wavfile


""" The following is only for test. """
# hp.set_hparam_yaml('default')
# dataset = LabelledDataset(hp.train.batch_size, hp.train.tar_path, hp.train.ntar_path, hp.signal.length)
# iterator = dataset().make_one_shot_iterator()
#
# with tf.Session() as sess:
#     next = iterator.get_next()
#     wav, melspec, label = sess.run(next)