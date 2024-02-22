from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import logging
from src.systems.color_bar_segmentation_system import ColorBarSegmentationSystem
from src.datasets.color_bar_segmentation_dataset import ColorBarSegmentationDataset

# Utility for using an existing model to make segmentation predictions about images
class ImageSegmenter:
  def __init__(self, config):
    self.config = config
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.load_model()

  def load_model(self):
    self.model = ColorBarSegmentationSystem.load_from_checkpoint(self.config.model_path, map_location = self.device)
    self.model.eval()

  # path - path to image to make predictions on, generally a resized version of the image
  # returns [predicted class, confidence]
  def predict(self, path):
    image_data = ColorBarSegmentationDataset.normalize_image(path, self.config.max_dimension)
    # Turn the image into a batch of one image
    image_batch = torch.unsqueeze(image_data, 0)

    outs = self.model(image_batch)
    top_predicted = [ColorBarSegmentationSystem.get_top_predicted(self.config, o) for o in outs]
    top_scores = ColorBarSegmentationSystem.get_top_scores(outs)

    return top_predicted[0], top_scores[0]
