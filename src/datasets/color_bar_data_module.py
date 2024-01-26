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
    val_size = int(num_images * config.val_percent)
    test_size = int(num_images * config.test_percent)
    test_end = val_size + test_size
    val_paths = list(Subset(image_paths, indices[0:val_size]))
    test_paths = list(Subset(image_paths, indices[val_size:test_end]))
    train_paths = list(Subset(image_paths, indices[test_end:]))

    self.val_dataset = config.dataset_class(config, val_paths, split='val')
    self.test_dataset = config.dataset_class(config, test_paths, split='test')
    self.train_dataset = config.dataset_class(config, train_paths, split='train')
    self.collate_fn = config.dataset_class.collate_fn
    print(f'Collate_fn {self.collate_fn}')

    self.batch_size = config.batch_size
    self.num_workers = config.num_workers

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size,
      shuffle = True, num_workers = self.num_workers, collate_fn = self.collate_fn, persistent_workers=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers, collate_fn = self.collate_fn, persistent_workers=True)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers, collate_fn = self.collate_fn, persistent_workers=True)
