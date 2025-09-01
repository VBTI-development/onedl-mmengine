<div align="center">
  <img width="600" alt="onedl-mmengine" src="https://mmassets.onedl.ai/assets/mmengine-logo.png" />
  <div>&nbsp;</div>
  <div align="center">
    <a href="https://vbti.ai">
      <b><font size="5">VBTI Website</font></b>
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://onedl.ai">
      <b><font size="5">OneDL platform</font></b>
    </a>
  </div>
<div>&nbsp;</div>

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://onedl-mmengine.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/vbti-development/onedl-mmengine.svg)](https://github.com/vbti-development/onedl-mmengine/blob/main/LICENSE)

[![pytorch](https://img.shields.io/badge/pytorch-2.0~2.5-yellow)](#installation)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onedl-mmengine)](https://pypi.org/project/onedl-mmengine/)
[![PyPI](https://img.shields.io/pypi/v/onedl-mmengine)](https://pypi.org/project/onedl-mmengine)

[![Build Status](https://github.com/VBTI-development/onedl-mmpretrain/actions/workflows/merge_stage_test.yml/badge.svg)](https://github.com/VBTI-development/onedl-mmpretrain/actions/workflows/merge_stage_test.yml)
[![open issues](https://isitmaintained.com/badge/open/VBTI-development/onedl-mmengine.svg)](https://github.com/VBTI-development/onedl-mmengine/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/VBTI-development/onedl-mmengine.svg)](https://github.com/VBTI-development/onedl-mmengine/issues)

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[📘Documentation](https://onedl-mmengine.readthedocs.io/en/latest/) |
[🤔Reporting Issues](https://github.com/vbti-development/onedl-mmengine/issues/new/choose)

</div>

## What's New

The VBTI development team is reviving MMLabs code, making it work with
newer pytorch versions and fixing bugs. We are only a small team, so your help
is appreciated. Also: since we don't speak or read Chinese, Chinese docs are deleted.

v0.10.6 was released on 2025-01-13.

Highlights:

- Support custom `artifact_location` in MLflowVisBackend [#1505](#1505)
- Enable `exclude_frozen_parameters` for `DeepSpeedEngine._zero3_consolidated_16bit_state_dict` [#1517](#1517)

Read [Changelog](./docs/en/notes/changelog.md#v0104-2342024) for more details.

## Introduction

MMEngine is a foundational library for training deep learning models based on PyTorch. It serves as the training engine of all OpenMMLab codebases, which support hundreds of algorithms in various research areas. Moreover, MMEngine is also generic to be applied to non-OpenMMLab projects. Its highlights are as follows:

**Integrate mainstream large-scale model training frameworks**

- [ColossalAI](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/large_model_training.html#colossalai)
- [DeepSpeed](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/large_model_training.html#deepspeed)
- [FSDP](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/large_model_training.html#fullyshardeddataparallel-fsdp)

**Supports a variety of training strategies**

- [Mixed Precision Training](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/speed_up_training.html#mixed-precision-training)
- [Gradient Accumulation](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/save_gpu_memory.html#gradient-accumulation)
- [Gradient Checkpointing](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/save_gpu_memory.html#gradient-checkpointing)

**Provides a user-friendly configuration system**

- [Pure Python-style configuration files, easy to navigate](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta)
- [Plain-text-style configuration files, supporting JSON and YAML](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html)

**Covers mainstream training monitoring platforms**

- [TensorBoard](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#tensorboard) | [WandB](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#wandb) | [MLflow](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#mlflow-wip)
- [ClearML](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#clearml) | [Neptune](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#neptune) | [DVCLive](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#dvclive) | [Aim](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html#aim)

## Installation

<details>
<summary>Supported PyTorch Versions</summary>

| MMEngine | PyTorch      | Python          |
| -------- | ------------ | --------------- |
| main     | >=1.6 \<=2.1 | >=3.10, \<=3.11 |

</details>

Before installing MMEngine, please ensure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/get-started/locally/).

Install MMEngine

```bash
pip install -U onedl-mim
mim install onedl-mmengine
```

Verify the installation

```bash
python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
```

## Get Started

Taking the training of a ResNet-50 model on the CIFAR-10 dataset as an example, we will use MMEngine to build a complete, configurable training and validation process in less than 80 lines of code.

<details>
<summary>Build Models</summary>

First, we need to define a **model** which 1) inherits from `BaseModel` and 2) accepts an additional argument `mode` in the `forward` method, in addition to those arguments related to the dataset.

- During training, the value of `mode` is "loss", and the `forward` method should return a `dict` containing the key "loss".
- During validation, the value of `mode` is "predict", and the forward method should return results containing both predictions and labels.

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

</details>

<details>
<summary>Build Datasets</summary>

Next, we need to create **Dataset**s and **DataLoader**s for training and validation.
In this case, we simply use built-in datasets supported in TorchVision.

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))
val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

</details>

<details>
<summary>Build Metrics</summary>

To validate and test the model, we need to define a **Metric** called accuracy to evaluate the model. This metric needs to inherit from `BaseMetric` and implements the `process` and `compute_metrics` methods.

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # Save the results of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })
    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # Returns a dictionary with the results of the evaluated metrics,
        # where the key is the name of the metric
        return dict(accuracy=100 * total_correct / total_size)
```

</details>

<details>
<summary>Build a Runner</summary>

Finally, we can construct a **Runner** with previously defined `Model`, `DataLoader`, and `Metrics`, with some other configs, as shown below.

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    # a wrapper to execute back propagation and gradient update, etc.
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # set some training configs like epochs
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
```

</details>

<details>
<summary>Launch Training</summary>

```python
runner.train()
```

</details>

## Learn More

<details>
<summary>Tutorials</summary>

- [Runner](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/runner.html)
- [Dataset and DataLoader](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/dataset.html)
- [Model](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/model.html)
- [Evaluation](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/evaluation.html)
- [OptimWrapper](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html)
- [Parameter Scheduler](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html)
- [Hook](https://onedl-mmengine.readthedocs.io/en/latest/tutorials/hook.html)

</details>

<details>
<summary>Advanced tutorials</summary>

- [Registry](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html)
- [Config](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html)
- [BaseDataset](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html)
- [Data Transform](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/data_transform.html)
- [Weight Initialization](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/initialize.html)
- [Visualization](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html)
- [Abstract Data Element](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html)
- [Distribution Communication](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/distributed.html)
- [Logging](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/logging.html)
- [File IO](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/fileio.html)
- [Global manager (ManagerMixin)](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/manager_mixin.html)
- [Use modules from other libraries](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/cross_library.html)
- [Test Time Agumentation](https://onedl-mmengine.readthedocs.io/en/latest/advanced_tutorials/test_time_augmentation.html)

</details>

<details>
<summary>Examples</summary>

- [Train a GAN](https://onedl-mmengine.readthedocs.io/en/latest/examples/train_a_gan.html)

</details>

<details>
<summary>Common Usage</summary>

- [Resume Training](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/resume_training.html)
- [Speed up Training](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/speed_up_training.html)
- [Save Memory on GPU](https://onedl-mmengine.readthedocs.io/en/latest/common_usage/save_gpu_memory.html)

</details>

<details>
<summary>Design</summary>

- [Hook](https://onedl-mmengine.readthedocs.io/en/latest/design/hook.html)
- [Runner](https://onedl-mmengine.readthedocs.io/en/latest/design/runner.html)
- [Evaluation](https://onedl-mmengine.readthedocs.io/en/latest/design/evaluation.html)
- [Visualization](https://onedl-mmengine.readthedocs.io/en/latest/design/visualization.html)
- [Logging](https://onedl-mmengine.readthedocs.io/en/latest/design/logging.html)
- [Infer](https://onedl-mmengine.readthedocs.io/en/latest/design/infer.html)

</details>

<details>
<summary>Migration guide</summary>

- [Migrate Runner from MMCV to MMEngine](https://onedl-mmengine.readthedocs.io/en/latest/migration/runner.html)
- [Migrate Hook from MMCV to MMEngine](https://onedl-mmengine.readthedocs.io/en/latest/migration/hook.html)
- [Migrate Model from MMCV to MMEngine](https://onedl-mmengine.readthedocs.io/en/latest/migration/model.html)
- [Migrate Parameter Scheduler from MMCV to MMEngine](https://onedl-mmengine.readthedocs.io/en/latest/migration/param_scheduler.html)
- [Migrate Data Transform to OpenMMLab 2.0](https://onedl-mmengine.readthedocs.io/en/latest/migration/transform.html)

</details>

## Contributing

We appreciate all contributions to improve MMEngine. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Citation

If you find this project useful in your research, please consider cite:

```
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/vbti-development/onedl-mmengine}},
  year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in VBTI-development

- [MMEngine](https://github.com/vbti-development/onedl-mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/vbti-development/onedl-mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/vbti-development/onedl-mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMDetection](https://github.com/vbti-development/onedl-mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMRotate](https://github.com/vbti-development/onedl-mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/vbti-development/onedl-mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMDeploy](https://github.com/vbti-development/onedl-mmdeploy): OpenMMLab model deployment framework.
- [MIM](https://github.com/vbti-development/onedl-mim): MIM installs OpenMMLab packages.

## Projects in OpenMMLab

- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
