from numpy import zeros
from src.datasets.color_bar_dataset import ColorBarDataset
from src.datasets.color_bar_segmentation_dataset import ColorBarSegmentationDataset
from src.utils.resnet_utils import load_for_resnet, load_mask_for_resnet
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_area
import torch
from torchvision import transforms

from PIL import Image

# Dataset class for predicting the part of an image containing the subject, without colorbars
# Returns image, target tuples as two tensors:
# A normalized image of (3, h, w) and a mask of (1, h, w)
# Where h and w are image dimensions after being resized for resnet
class SubjectSegmentationDataset(ColorBarSegmentationDataset):
  def __init__(self, config, image_paths, split = 'train'):
    self.boxes = []
    super().__init__(config, image_paths, split)

  # Loads annotation data into self.labels in the same order they paths are listed in image_paths
  def load_labels(self, path_to_labels):
   # The label must be a mask rather than a binary [0,1] class 
    for index, image_path in enumerate(self.image_paths):
      if not image_path:
        continue
      # Add image dimensions
      w, h = self.config.max_dimension, self.config.max_dimension
      image_labels = path_to_labels[str(image_path)]
      bounding_boxes = []
      bar_box = None
      self.labels.append([1])
      has_color_bar = any(entry.get("rectanglelabels") == ["color_bar"] for entry in image_labels)
      for label in image_labels:
        if 'subject' in label['rectanglelabels']:
          norm_x, norm_y, norm_x2, norm_y2 = self.round_box_to_edge(label, has_color_bar)
          x1, y1, x2, y2 = self.norms_to_pixels(norm_x, norm_y, norm_x2, norm_y2, w, h)
          bar_box = [x1, y1, x2, y2]
          bounding_boxes.append(bar_box)
      
      if len(bounding_boxes) == 0:
        bounding_boxes.append([0, 0, w, h])
      self.boxes.append(torch.tensor(bounding_boxes, dtype=torch.float32))

  def round_box_to_edge(self, label, has_color_bar):
    rounded = super().round_box_to_edge(label)
    if has_color_bar and rounded == (0, 0, 1.0, 1.0):
      x, y = label['x'], label['y']
      x2, y2 = x + label['width'], y + label['height']
      return (x / 100, y / 100, x2 / 100, y2 / 100)
    return rounded