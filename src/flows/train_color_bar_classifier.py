import os
import torch
import random
import numpy as np

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.training_config import TrainingConfig
from src.datasets import ColorBarDataModule
from src.systems import ColorBarClassifyingSystem

class TrainColorBarClassifier(FlowSpec):
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

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config

    self.next(self.train_test)

  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'pre-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_baseline.py`. To list
  this flow, run `python flow_baseline.py show`. To execute
  this flow, run `python flow_baseline.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_baseline.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_baseline.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainBaseline()