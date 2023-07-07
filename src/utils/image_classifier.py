from PIL import Image
from pathlib import Path
import torch
import logging
from src.utils.resnet_utils import load_for_resnet, resnet50_foundation_model

# Utility for using an existing model to make predictions about images
class ImageClassifier:
  def __init__(self, config):
    self.config = config
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def load_model(self):
    # Load configured model
    self.foundation_model = resnet50_foundation_model(self.device)
    self.foundation_model.eval()
    self.model = nn.Sequential(
      nn.Linear(starting_size, self.config.model_width),
      nn.ReLU(inplace=True),
      nn.Linear(self.config.model_width, 1),
    ).to(self.device)
    self.model.load_state_dict(torch.load(self.config.model_path))
    self.model.eval()

  # path - path to image to make predictions on, generally a resized version of the image
  # returns [predicted class, confidence]
  def predict(self, path):
    image_data = load_for_resnet(path, self.config.max_dimension)

    with torch.no_grad():
      fdn_output = self.foundation_model(image_data).flatten(1)
      logits = self.model(fdn_output)
      raw_predictions = torch.sigmoid(logits)
      predicted_classes = torch.where(raw_predictions > self.config.predict_rounding_threshold, 1, 0)
      return raw_predictions, predicted_classes
