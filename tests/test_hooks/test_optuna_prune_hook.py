# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.hooks.optuna_prune_hook import OptunaPruneHook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmengine.testing import RunnerTestCase


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class DummyMetric(BaseMetric):

    default_prefix: str = 'test'

    def __init__(self, length):
        super().__init__()
        self.length = length
        self.best_idx = length // 2
        self.cur_idx = 0
        self.vals = [90, 91, 92, 88, 89, 90] * 2

    def process(self, *args, **kwargs):
        self.results.append(0)

    def compute_metrics(self, *args, **kwargs):
        acc = self.vals[self.cur_idx]
        self.cur_idx += 1
        return dict(acc=acc)


class DummyTrial:

    def __init__(self):
        self.values = []

    def report(self, value, step):
        self.values.append((step, value))

    def should_prune(self):
        if len(self.values) < 3:
            return False
        return True


def get_mock_runner():
    runner = Mock()
    runner.train_loop = Mock()
    runner.train_loop.stop_training = False
    runner.epoch = 0
    return runner


class TestOptunaPruneHook(RunnerTestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.optuna_study = None
        self.optuna_trial = DummyTrial()

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_init(self):
        hook = OptunaPruneHook(trial=self.optuna_trial, monitor='acc')
        self.assertEqual(hook.monitor, 'acc')

    def test_before_run(self):
        runner = Mock()
        runner.train_loop = object()

        # `train_loop` must contain `stop_training` variable.
        with self.assertRaises(AssertionError):
            hook = OptunaPruneHook(
                trial=self.optuna_trial, monitor='accuracy/top1')
            hook.before_run(runner)

    def test_after_val_epoch(self):
        runner = get_mock_runner()
        metrics = {'accuracy/top1': 0.5, 'loss': 0.23}
        hook = OptunaPruneHook(trial=self.optuna_trial, monitor='acc')

        with self.assertWarns(UserWarning):
            # Skip early stopping process since the evaluation results does not
            # include the key 'acc'
            hook.after_val_epoch(runner, metrics)

        # if `monitor` does not match and strict=True, crash the training.
        with self.assertRaises(RuntimeError):
            metrics = {'accuracy/top1': 0.5, 'loss': 0.23}
            hook = OptunaPruneHook(
                trial=self.optuna_trial, monitor='acc', strict=True)
            hook.after_val_epoch(runner, metrics)

        # Check largest value
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i / 9.} for i in range(8)]
        hook = OptunaPruneHook(
            trial=self.optuna_trial, monitor='accuracy/top1')
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
            if runner.train_loop.stop_training:
                break
        self.assertAlmostEqual(metric['accuracy/top1'], 2 / 9)
        self.assertTrue(runner.train_loop.stop_training)

    def test_with_runner(self):
        max_epoch = 10
        work_dir = osp.join(self.temp_dir.name, 'runner_test')
        optuna_prune_cfg = dict(
            type='OptunaPruneHook',
            trial=self.optuna_trial,
            monitor='test/acc',
        )
        runner = Runner(
            model=ToyModel(),
            work_dir=work_dir,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=dict(type=DummyMetric, length=max_epoch),
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(
                by_epoch=True, max_epochs=max_epoch, val_interval=1),
            val_cfg=dict(),
            custom_hooks=[optuna_prune_cfg],
            experiment_name='optunaprune_test')
        runner.train()
        self.assertEqual(runner.epoch, 3)
        self.assertTrue(runner.train_loop.stop_training)
