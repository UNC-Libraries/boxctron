import os
import torch
import random
import numpy as np
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.training_config import TrainingConfig
from src.datasets import ColorBarDataModule
from src.systems import ColorBarClassifyingSystem
from src.utils.json_utils import to_json

LOG_DIR = Path('logs').resolve()

class TrainColorBarClassifier(FlowSpec):
  config_path = Parameter('config', 
    help = 'path to config file', default = Path('fixtures/test_config.json'))

  @step
  def start(self):
    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = TrainingConfig(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = ColorBarDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = ColorBarClassifyingSystem(config)
    print(f"Initializing system, saving to {config.save_dir}")

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.save_dir,
      monitor = 'val_loss',
      mode = 'min',    # look for lowest `val_loss`
      save_last = True,
      save_top_k = 1,  # save top 1 checkpoints
      every_n_epochs=1,
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.max_epochs,
      enable_progress_bar = config.enable_progress_bar,
      logger = TensorBoardLogger(save_dir=config.log_dir),
      callbacks = [checkpoint_callback])

    # Store variables for passing to the next step
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config

    self.next(self.train_model)

  @step
  def train_model(self):
    self.trainer.fit(self.system, self.dm)
    self.next(self.offline_test)

  @step
  def offline_test(self):
    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = self.config.log_dir / 'results.json'
    os.makedirs(str(log_file.parent), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('Training completed')


if __name__ == "__main__":
  """
  To validate this flow, run `python train_color_bar_classifier.py`. To list
  this flow, run `python train_color_bar_classifier.py show`. To execute
  this flow, run `python train_color_bar_classifier.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python train_color_bar_classifier.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python train_color_bar_classifier.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainColorBarClassifier()