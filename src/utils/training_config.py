from pathlib import Path
import json
import importlib
from src.datasets.color_bar_classifying_dataset import ColorBarClassifyingDataset

class TrainingConfig:
  def __init__(self, path):
    with open(path) as json_data:
      data = json.load(json_data)
      # Path to file which lists images in the training set
      self.image_list_path = Path(data['image_list_path'])
      # Path to json file containing all the training annotations
      self.annotations_path = Path(data['annotations_path'])
      # Base path where image paths should be evaluated against
      self.base_image_path = Path(data.get('base_image_path', '/'))
      # dataset class to use to load the training data
      self.dataset_class = data.get('dataset_class', ColorBarClassifyingDataset)
      if isinstance(self.dataset_class, str):
        ds_module, ds_class = self.dataset_class.rsplit(".", 1)
        ds_module = importlib.import_module(ds_module)
        self.dataset_class = getattr(ds_module, ds_class)
      # Size of batches to use in training
      self.batch_size = data.get('batch_size', 8)
      # Number of workers to use when training
      self.num_workers = data.get('num_workers', 4)
