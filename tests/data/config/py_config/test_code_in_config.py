# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import pathlib

from mmengine import Config  # isort:skip

# Get the path of the current file, even if __file__ is not defined
current_file = pathlib.Path(inspect.getfile(inspect.currentframe()))
cfg = Config.fromfile(current_file.parent / 'simple_config.py')
item5 = cfg.item1[0] + cfg.item2.a
