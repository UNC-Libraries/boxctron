import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from src.utils.resnet_utils import load_for_resnet
from src.utils.json_utils import from_json

# Base dataset class for color bar data
class ColorBarDataset(torch.utils.data.Dataset):
  def __init__(self, config, split = 'train'):
    super().__init__()
    self.config = config
    self.labels = []
    self.split = split
    # Load the list of files
    with open(self.config.image_list_path) as f:
      self.image_paths = [Path(p).resolve() for p in f.read().splitlines()]
    # Load the annotations from label studio
    annotations = from_json(self.config.annotations_path)
    # build a map of image file paths to label info
    path_to_labels = {}
    for anno in annotations:
      img_path = (self.config.base_image_path / anno['image'].split('/', 3)[3]).resolve()
      path_to_labels[str(img_path)] = anno['label']
    # Load labels from the annotation mapping
    self.load_labels(path_to_labels)

  # Loads annotation data into self.labels in the same order they paths are listed in image_paths
  def load_labels(self, path_to_labels):
    raise Exception("Not implemented for parent class")

  def __getitem__(self, index):
    image_data = load_for_resnet(self.image_paths[index])
    return image_data, self.labels[index]
    
  def __len__(self):
    return len(self.image_paths)