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
      self.base_image_path = Path(data.get('base_image_path', '/')).resolve()
      # dataset class to use to load the training data
      self.dataset_class = data.get('dataset_class', ColorBarClassifyingDataset)
      if isinstance(self.dataset_class, str):
        ds_module, ds_class = self.dataset_class.rsplit(".", 1)
        ds_module = importlib.import_module(ds_module)
        self.dataset_class = getattr(ds_module, ds_class)
      # Data module/optimizer settings
      # Size of batches to use in training
      self.batch_size = data.get('batch_size', 8)
      # Number of workers to use when training
      self.num_workers = data.get('num_workers', 4)
      # Number of epoches to train over
      self.max_epochs = data.get('max_epochs', 3)
      # learning rate, how quickly the model will decide it is right during quickly. Higher the number, faster it goes.
      self.lr = data.get('lr', 3e-4)
      # Directory where training checkpoints will be saved
      self.save_dir = data.get('save_dir', './artifacts/ckpts/train')
      # Max dimension size which images will be normalized to
      self.max_dimension = data.get('max_dimension', 1333)
      # Percentage of the dataset which will be used for evaluating the model during development/training
      self.val_percent = data.get('val_percent', 0.2)
      # Percentage of the dataset which will be used for evaluating the model after training completes
      self.test_percent = data.get('test_percent', 0)
    # Percentage of the dataset which will be used for the training dataset, which is the remainder not used for dev or test.
    self.train_percent = 1 - self.val_percent - self.test_percent

