from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import logging
from src.systems.color_bar_classifying_system import ColorBarClassifyingSystem
from src.utils.resnet_utils import load_for_resnet

# Utility for using an existing model to make predictions about images
class ImageClassifier:
  def __init__(self, config):
    self.config = config
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.load_model()

  def load_model(self):
    self.model = ColorBarClassifyingSystem.load_from_checkpoint(self.config.model_path, map_location = self.device)
    self.model.eval()

  # path - path to image to make predictions on, generally a resized version of the image
  # returns [predicted class, confidence]
  def predict(self, path):
    image_data = load_for_resnet(path, self.config.max_dimension)
    # Turn the image into a batch of one image
    image_batch = torch.unsqueeze(image_data, 0)

    logits = self.model(image_batch)
    raw_predictions = torch.sigmoid(logits)
    predicted_classes = torch.where(raw_predictions > self.config.predict_rounding_threshold, 1, 0)
    return raw_predictions, predicted_classes
