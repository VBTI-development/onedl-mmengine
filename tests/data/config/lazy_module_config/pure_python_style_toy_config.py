# Copyright (c) VBTI. All rights reserved.
from torch.optim import SGD
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook,
                            IterTimerHook, LoggerHook, ParamSchedulerHook,
                            RuntimeInfoHook)
from mmengine.model import MMDistributedDataParallel
from mmengine.optim import MultiStepLR, OptimWrapper
import importlib.util
import os

# Dynamically load the test module by file path to avoid relative import
# issues when the config is parsed during tests.
_mod_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test_runner', 'test_runner.py'))
spec = importlib.util.spec_from_file_location('test_runner_testmod', _mod_path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
ToyDataset = _mod.ToyDataset
ToyModel = _mod.ToyModel
ToyMetric1 = _mod.ToyMetric1

# Clean up temporary variables
del importlib
del os
del spec
del _mod
del _mod_path

model=dict(type=ToyModel)
train_dataloader=dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_size=3,
    num_workers=0)
val_dataloader=dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=False),
    batch_size=3,
    num_workers=0)
test_dataloader=dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=False),
    batch_size=3,
    num_workers=0)
auto_scale_lr=dict(base_batch_size=16, enable=False)
optim_wrapper=dict(
    type=OptimWrapper, optimizer=dict(type=SGD, lr=0.01))
model_wrapper_cfg=dict(type=MMDistributedDataParallel)
param_scheduler=dict(type=MultiStepLR, milestones=[1, 2])
val_evaluator=dict(type=ToyMetric1)
test_evaluator=dict(type=ToyMetric1)
train_cfg=dict(
    by_epoch=True, max_epochs=3, val_interval=1, val_begin=1)
val_cfg=dict()
test_cfg=dict()
custom_hooks=[]
default_hooks=dict(
    runtime_info=dict(type=RuntimeInfoHook),
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook, interval=1, by_epoch=True),
    sampler_seed=dict(type=DistSamplerSeedHook))
data_preprocessor=None
launcher = 'pytorch'
env_cfg=dict(dist_cfg=dict(backend='nccl'))
