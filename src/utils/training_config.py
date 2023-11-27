from pathlib import Path
import json
import importlib
from src.datasets.color_bar_classifying_dataset import ColorBarClassifyingDataset
from src.systems.color_bar_classifying_system import ColorBarClassifyingSystem

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
      # class for the system used for training the model
      self.system_class = self.get_class_property(data, 'system_class', ColorBarClassifyingSystem)
      # dataset class to use to load the training data
      self.dataset_class = self.get_class_property(data, 'dataset_class', ColorBarClassifyingDataset)
      # Data module/optimizer settings
      # Size of batches to use in training
      self.batch_size = data.get('batch_size', 8)
      # Number of workers to use when training
      self.num_workers = data.get('num_workers', 4)
      # Number of epoches to train over
      self.max_epochs = data.get('max_epochs', 3)
      # learning rate, how quickly the model will decide it is right during quickly. Higher the number, faster it goes.
      self.lr = data.get('lr', 3e-4)
      # Weight decay for l2 regularization
      self.weight_decay = data.get('weight_decay', 1e-5)
      # Directory where training checkpoints will be saved
      self.save_dir = data.get('save_dir', './artifacts/ckpts/train')
      # Max dimension size which images will be normalized to
      self.max_dimension = data.get('max_dimension', 1333)
      # Percentage of the dataset which will be used for evaluating the model during development/training
      self.val_percent = data.get('val_percent', 0.2)
      # Percentage of the dataset which will be used for evaluating the model after training completes
      self.test_percent = data.get('test_percent', 0)
      # Whether or not to show the progress bar during training
      self.enable_progress_bar = data.get('enable_progress_bar', False)
      # Directory where progress and results will be saved to
      self.log_dir = Path(data.get('log_dir', './logs/')).resolve()
      # Width of the hidden layer used in between the foundation model and the file result
      self.model_width = data.get('model_width', 256)
      # Depth of the resnet foundation model to load for transfer learning
      self.resnet_depth = data.get('resnet_depth', 50)
      # How often metrics will get logged during training
      self.log_every_n_steps = data.get('log_every_n_steps', 10)
      # Only predictions with confidence higher than this threshold will be counted as matching the class
      self.predict_rounding_threshold = data.get('predict_rounding_threshold', 0.7)
      # Metric that will be used to evaluate how well an epoch has performed
      self.validation_monitor_metric = data.get('validation_monitor_metric', 'val_fp_loss')
      # How to compare metric values across epochs, 'min' means lower values are better
      self.validation_monitor_mode = data.get('validation_monitor_mode', 'min')
    # Percentage of the dataset which will be used for the training dataset, which is the remainder not used for dev or test.
    self.train_percent = 1 - self.val_percent - self.test_percent

  def get_class_property(self, data, prop_name, default_class):
    class_val = data.get(prop_name, default_class)
    if isinstance(class_val, str):
      class_module, class_name = class_val.rsplit(".", 1)
      class_module = importlib.import_module(class_module)
      class_val = getattr(class_module, class_name)
    return class_val

