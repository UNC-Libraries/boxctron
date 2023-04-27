import os
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from pprint import pprint
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.training_config import TrainingConfig
from src.datasets import ColorBarDataModule
from src.systems import ColorBarClassifyingSystem
from src.utils.json_utils import to_json

class TrainColorBarClassifier:
  def init_system(self, config_path):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    self.config = TrainingConfig(config_path)

    # a data module wraps around training, dev, and test datasets
    self.dm = ColorBarDataModule(self.config)

    # a PyTorch Lightning system wraps around model logic
    self.system = ColorBarClassifyingSystem(self.config)
    self.log(f'Initializing system, saving to {self.config.save_dir}')

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = self.config.save_dir,
      monitor = 'val_loss',
      mode = 'min',    # look for lowest `val_loss`
      save_last = True,
      save_top_k = 1,  # save top 1 checkpoints
      every_n_epochs=1,
      verbose = True,
    )

    self.trainer = Trainer(
      max_epochs = self.config.max_epochs,
      log_every_n_steps = self.config.log_every_n_steps,
      enable_progress_bar = self.config.enable_progress_bar,
      logger = TensorBoardLogger(save_dir=self.config.log_dir),
      callbacks = [checkpoint_callback])

  def train_model(self):
    self.log('Training model')
    self.trainer.fit(self.system, self.dm)

  def offline_test(self):
    self.log('Testing model')
    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = self.config.log_dir / 'results.json'
    os.makedirs(str(log_file.parent), exist_ok = True)
    to_json(results, log_file)  # save to disk
    self.log('Training completed')

  def log(self, message):
    print(f'{datetime.now().isoformat()} {message}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train color bar classifier.')
  parser.add_argument('-c', '--config', dest='config', type=str,
                    default='fixtures/test_config.json',
                    help='Path to training config')
  args = parser.parse_args()

  train_classifier = TrainColorBarClassifier()
  train_classifier.init_system(args.config)
  train_classifier.train_model()
  train_classifier.offline_test()