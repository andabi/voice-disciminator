# -*- coding: utf-8 -*-
#!/usr/bin/env python

""" Runnable code. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import fire
import tensorflow as tf
from tensorpack.callbacks.base import Callback
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.input_source.input_source import TFDatasetInput
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated, SimpleTrainer
from tensorpack.utils import logger

from data import LabelledDataset, UnLabelledDataset
from hparam import hparam as hp
from models import default_model
from utils import remove_all_files


class EvalCallback(Callback):

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['melspec', 'labels'], ['loss', 'accuracy'])
        self.dataset = LabelledDataset(hp.eval.batch_size, hp.eval.tar_path, hp.eval.ntar_path, length=hp.signal.length, tar_ratio=0.5)

    def _trigger_epoch(self):
        wav, melspec, label = zip(*list(self.dataset.get_random_wav_and_label() for _ in range(hp.eval.batch_size)))
        loss, acc = self.pred(melspec, label)
        self.trainer.monitors.put_scalar('eval/loss', loss)
        self.trainer.monitors.put_scalar('eval/accuracy', acc)


class Runner(object):

    def train(self, case='default', ckpt=None, gpu=None, r=False):
        """
        :param case: experiment case name
        :param ckpt: checkpoint to load model
        :param gpu: comma separated list of GPU(s) to use
        :param r: start from the beginning.
        """
        hp.set_hparam_yaml(case)
        if r:
            remove_all_files(hp.logdir)

        # dataset
        dataset = LabelledDataset(hp.train.batch_size, hp.train.tar_path, hp.train.ntar_path, length=hp.signal.length, tar_ratio=hp.train.tar_ratio)

        # set logger for event and model saver
        logger.set_logger_dir(hp.logdir)

        train_conf = TrainConfig(
            model=default_model,
            data=TFDatasetInput(dataset()),
            callbacks=[
                ModelSaver(checkpoint_dir=hp.logdir),
                EvalCallback(),
                # RunUpdateOps()  # enable this when using batch normalization.
            ],
            max_epoch=hp.train.num_epochs,
            steps_per_epoch=hp.train.steps_per_epoch,
        )
        ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
        if ckpt:
            train_conf.session_init = SaverRestore(ckpt)

        if gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu))
            train_conf.nr_tower = len(gpu)

        if hp.train.num_gpu <= 1:
            trainer = SimpleTrainer()
        else:
            trainer = SyncMultiGPUTrainerReplicated(gpus=hp.train.num_gpu)

        launch_train_with_config(train_conf, trainer=trainer)

    def discriminate(self, case='default', model=default_model, ckpt=None, gpu=None):
        """
        :param case: experiment case name
        :param ckpt: checkpoint to load model
        :param gpu: comma separated list of GPU(s) to use
        """

        hp.set_hparam_yaml(case)

        # dataset
        dataset = UnLabelledDataset(hp.disc.batch_size, hp.disc.data_path, length=hp.signal.length)

        # sample
        iterator = dataset().make_one_shot_iterator()
        wav_tensor, melspec_tensor, wavfile_tensor = iterator.get_next()

        # feed forward
        logit_tensor, pred_tensor = model.discriminate(melspec_tensor, is_training=False)
        prob_tensor = tf.nn.softmax(logit_tensor)

        # summaries
        # summ_op = tf.summary.merge_all()

        session_config = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 1},
        )
        with tf.Session(config=session_config) as sess:

            # load model
            ckpt = '{}/{}'.format(hp.logdir, ckpt) if ckpt else tf.train.latest_checkpoint(hp.logdir)
            sess.run(tf.global_variables_initializer())
            if ckpt:
                tf.train.Saver().restore(sess, ckpt)
                print('Successfully loaded checkpoint {}'.format(ckpt))
            else:
                print('No checkpoint found at {}.'.format(hp.logdir))

            # classification
            while True:
                try:
                    pred, wavfile, prob, logit = sess.run([pred_tensor, wavfile_tensor, prob_tensor, logit_tensor])
                    # tar_wavfile, tar_prob = zip(*[(w, pr) for p, w, pr in zip(pred, wavfile, prob) if p == 1])
                    # print('target wavfile size: {}'.format(len(tar_wavfile)))
                    # for w, p in zip(tar_wavfile, tar_prob):
                    for w, p, pr in zip(wavfile, pred, prob):
                        print(w, p, pr)
                except tf.errors.OutOfRangeError:
                    break

            # summ = sess.run([summ_op])

        # write summaries in tensorboard
        writer = tf.summary.FileWriter(hp.logdir)
        # writer.add_summary(summ)
        writer.close()

        print('Done.')


if __name__ == '__main__':
    fire.Fire(Runner)