# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings

import optuna

from mmengine.logging.logger import print_log
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class OptunaPruneHook(Hook):
    """Early stop the training when Optuna prunes the trial.

    Either give the Optuna `trial` object directly or load it from storage by
    specifying `study_name`, `trial_number`, and `study_storage_url`.

    Args:
        monitor:
            The monitored metric key to report to optuna
        study_name:
            The name of the Optuna study.
        trial_number:
            The number of the Optuna trial.
        study_storage_url:
            The storage URL of the Optuna study.
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of
            the objective function.
        strict (bool, optional): Whether to crash the training when `monitor`
            is not found in the `metrics`. Defaults to False.
    Note:
        `New in version 0.11.0.`
    """

    def __init__(
        self,
        monitor: str,
        study_name: str | None = None,
        trial_number: int | None = None,
        study_storage_url: str | None = None,
        trial: None | optuna.trial.Trial = None,
        strict: bool = False,
    ) -> None:
        if trial is not None:
            print_log(
                'Using the provided Optuna trial.',
                logger='current',
                level=logging.INFO)
            self.trial = trial
        else:
            # load from storage
            if study_name is None or trial_number is None \
                    or study_storage_url is None:
                raise ValueError(
                    'study_name, trial_number, and study_storage_url must be '
                    'provided when trial is not given.')
            print_log(
                'Loading Optuna trial from storage.',
                logger='current',
                level=logging.INFO)
            self.trial = self._load_trial(
                study_name=study_name,
                trial_number=trial_number,
                study_storage_url=study_storage_url)
        self.monitor = monitor
        self.strict = strict

    def _load_trial(
        self,
        study_name: str,
        trial_number: int,
        study_storage_url: str,
    ) -> optuna.trial.Trial:
        """Load the Optuna trial from storage.

        Args:
            study_name: The name of the Optuna study.
            trial_number: The number of the Optuna trial.
            study_storage_url: The storage URL of the Optuna study.
        """
        storage = optuna.storages.RDBStorage(url=study_storage_url)
        study: optuna.Study = optuna.load_study(
            study_name=study_name, storage=storage)
        trial = study.trials[trial_number]

        return trial

    def before_run(self, runner) -> None:
        """Check `stop_training` variable in `runner.train_loop`.

        Args:
            runner (Runner): The runner of the training process.
        """

        assert hasattr(runner.train_loop, 'stop_training'), \
            '`train_loop` should contain `stop_training` variable.'

    def after_val_epoch(self, runner, metrics):
        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """
        if self.monitor not in metrics:
            if self.strict:
                raise RuntimeError(
                    'OptunaPruneHook stopping conditioned on metric '
                    f'`{self.monitor} is not available. Please check available'
                    f' metrics {metrics}, or set `strict=False` in '
                    '`OptunaPruneHook`.')
            warnings.warn(
                'Skip pruning process since the evaluation '
                f'results ({metrics.keys()}) do not include `monitor` '
                f'({self.monitor}).')
            return

        self.trial.report(float(metrics[self.monitor]), step=runner.epoch)
        if self.trial.should_prune():
            runner.train_loop.stop_training = True
            message = (f'Trial was pruned at epoch {runner.epoch} '
                       f'with {self.monitor}={metrics[self.monitor]:.3f}.')
            runner.logger.info(message)
