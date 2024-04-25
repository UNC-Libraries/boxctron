from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import logging
from src.utils.segmentation_utils import get_top_predicted, get_top_scores
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

  # paths - paths of images to make predictions on, generally a resized version of the image
  # returns [list of predicted classes, list of confidences]
  def predict(self, paths):
    batch_data = torch.empty(len(paths), 3, self.config.max_dimension, self.config.max_dimension)
    for i, path in enumerate(paths):
        image_data = ColorBarSegmentationDataset.normalize_image(path, self.config.max_dimension)
        batch_data[i] = image_data

    outs = self.model(batch_data)
    top_predicted = [get_top_predicted(self.config.predict_rounding_threshold, o) for o in outs]
    top_scores = get_top_scores(outs)

    return top_predicted, top_scores
