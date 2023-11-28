import os
from pathlib import Path
from pprint import pprint
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.training_config import TrainingConfig
from src.datasets.color_bar_data_module import ColorBarDataModule
from src.systems.color_bar_classifying_system import ColorBarClassifyingSystem
from src.utils.json_utils import to_json
from src.utils.common_utils import log

class ColorBarModelTrainer:
  def init_system(self, config_path):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    self.config = TrainingConfig(config_path)

    # a data module wraps around training, dev, and test datasets
    self.dm = ColorBarDataModule(self.config)

    # a PyTorch Lightning system wraps around model logic
    self.system = self.config.system_class(self.config)
    log(f'Initializing system, saving to {self.config.save_dir}')

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = self.config.save_dir,
      monitor = self.config.validation_monitor_metric,
      mode = self.config.validation_monitor_mode,
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
    self.trainer.fit(self.system, self.dm)

  def validation_evaluation(self):
    self.trainer.validate(self.system, self.dm, ckpt_path = 'best')

  def get_validation_incorrect_results(self):
    return self.system.record_val_incorrect_predictions(self.dm.val_dataset)

  def get_validation_incorrect_results_as_csv(self):
    return self.get_validation_incorrect_results().to_csv(index=False)

  def offline_test(self):
    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

  def get_test_results(self):
    return self.system.test_results

  def write_test_results(self):
    log_file = self.config.log_dir / 'results.json'
    os.makedirs(str(log_file.parent), exist_ok = True)
    to_json(results, log_file)  # save to disk

  def get_test_incorrect_results(self):
    return self.system.record_test_incorrect_predictions(self.dm.test_dataset)

  def get_test_incorrect_results_as_csv(self):
    return self.get_test_incorrect_results().to_csv(index=False)