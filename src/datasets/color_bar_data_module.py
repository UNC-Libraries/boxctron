import pytorch_lightning as pl
import torch
from pathlib import Path
from torch.utils.data import Subset, DataLoader

class ColorBarDataModule(pl.LightningDataModule):
  def __init__(self, config):
    super().__init__()

    with open(config.image_list_path) as f:
      image_paths = [Path(p).resolve() for p in f.read().splitlines()]
    num_images = len(image_paths)
    indices = torch.randperm(num_images).tolist()
    dev_size = int(num_images * config.dev_percent)
    test_size = int(num_images * config.test_percent)
    test_end = dev_size + test_size
    dev_paths = list(Subset(image_paths, indices[0:dev_size]))
    test_paths = list(Subset(image_paths, indices[dev_size:test_end]))
    train_paths = list(Subset(image_paths, indices[test_end:]))

    self.dev_dataset = config.dataset_class(config, dev_paths, split='dev')
    self.test_dataset = config.dataset_class(config, test_paths, split='test')
    self.train_dataset = config.dataset_class(config, train_paths, split='train')

    self.batch_size = config.batch_size
    self.num_workers = config.num_workers

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size,
      shuffle = True, num_workers = self.num_workers)

  def dev_dataloader(self):
    return DataLoader(self.dev_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers)
