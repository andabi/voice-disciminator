# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob

import numpy as np
import tensorflow as tf

from audio import read_wav, wav2melspec_db, trim_wav, crop_random_wav
from hparam import hparam as hp


class Dataset():
    """
    A reference for best practices: 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/performance/datasets_performance.md#summary-of-best-practices
    """

    def __init__(self, batch_size, tar_path, ntar_path, length=4000, tar_ratio=0.5):
        self.batch_size = batch_size
        # self.target_path = tar_path
        # self.non_target_path = ntar_path
        self.tar_wavfiles = glob.glob(tar_path)
        self.ntar_wavfiles = glob.glob(ntar_path)
        self.length = length
        self.tar_ratio = tar_ratio

    def __call__(self, n_prefetch=1000, n_thread=32):
        dataset = tf.data.Dataset.from_tensors(0)
        dataset = dataset.map(
            lambda _: tf.py_func(self._get_random_wav_and_label, [], [tf.float32, tf.int64, tf.float64]),
            num_parallel_calls=n_thread)
        dataset = dataset.repeat().batch(self.batch_size).prefetch(n_prefetch)
        return dataset

    def _get_random_wav_and_label(self):
        """

        :return: wav: raw wave. float32. shape=(t, ),
                 label: 1 if target, 0 otherwise. int64.
                 melspec: mel-spectrogram. float64. shape=(t, n_mels)
        """
        wavfiles, label = (self.tar_wavfiles, 1) if np.random.sample(1) <= self.tar_ratio else (self.ntar_wavfiles, 0)
        wavfile = np.random.choice(wavfiles, 1)[0]
        wav = read_wav(wavfile, sr=hp.signal.sr)
        wav = trim_wav(wav)
        wav = crop_random_wav(wav, self.length)
        melspec = wav2melspec_db(wav, sr=hp.signal.sr, n_fft=hp.signal.n_fft, win_length=hp.signal.win_length,
                                 hop_length=hp.signal.hop_length, n_mels=hp.signal.n_mels,
                                 min_db=hp.signal.min_db, max_db=hp.signal.max_db)
        return wav, label, melspec


""" The following is only for test. """
hp.set_hparam_yaml('default')
dataset = Dataset(hp.train.batch_size, hp.train.tar_path, hp.train.ntar_path, hp.signal.length)
iterator = dataset().make_one_shot_iterator()

with tf.Session() as sess:
    next = iterator.get_next()
    wav, label, melspec = sess.run(next)