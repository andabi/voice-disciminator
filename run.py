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

from data import Dataset
from hparam import hparam as hp
from models import default_model
from utils import remove_all_files


class EvalCallback(Callback):

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['melspec', 'labels'], ['loss', 'accuracy'])
        self.dataset = Dataset(hp.eval.batch_size, hp.eval.data_path, hp.eval.data_path, length=hp.signal.length, tar_ratio=1.)

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
        dataset = Dataset(hp.train.batch_size, hp.train.tar_path, hp.train.ntar_path, length=hp.signal.length)

        # set logger for event and model saver
        logger.set_logger_dir(hp.logdir)

        train_conf = TrainConfig(
            model=default_model,
            data=TFDatasetInput(dataset()),
            callbacks=[
                ModelSaver(checkpoint_dir=hp.logdir),
                EvalCallback(),
                # RunUpdateOps()
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

if __name__ == '__main__':
    fire.Fire(Runner)