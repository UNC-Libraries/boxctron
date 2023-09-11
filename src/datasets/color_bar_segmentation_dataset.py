from numpy import zeros, uint8
from torch import max
from src.datasets.color_bar_dataset import ColorBarDataset
from src.utils.resnet_utils import load_for_resnet, load_mask_for_resnet
from PIL import Image

# Dataset with single class for labeling images as having a color bar or not
class ColorBarSegmentationDataset(ColorBarDataset):
  def __init__(self, config, image_paths, split = 'train'):
    super().__init__(config, image_paths, split)
  
  # Must be overriden from parent class
  def __getitem__(self, index):
    image_data = load_for_resnet(self.image_paths[index], self.config.max_dimension)
    label_mask = load_mask_for_resnet(self.labels[index], self.config.max_dimension)
    label_mask = label_mask.clamp(0,1)
    return image_data, label_mask

  # Loads annotation data into self.labels in the same order they paths are listed in image_paths
  def load_labels(self, path_to_labels):
   # The label must be a mask rather than a binary [0,1] class 
   for index, image_path in enumerate(self.image_paths):
     if not image_path:
       continue
    # Add image dimensions
     w, h = None, None
     with Image.open(image_path) as img:
      w, h = img.width, img.height
     self.image_dimensions.append((w, h))
     # Populate label masks
     image_labels = path_to_labels[str(image_path)]
     mask = zeros((h, w), dtype=uint8)
     for label in image_labels:
       if 'color_bar' in label['rectanglelabels']:
        width, height = (label['width']), (label['height'])
        x = int(label['x']/100 * label['original_width'])
        y = int(label['y']/100 * label['original_height'])
        x2 = x + int(width/100 * label['original_width']) # bar width
        y2 = y + int(height/100 * label['original_height']) # bar height
        mask[y:y2, x:x2] = 1 # Mark all pixels in the masked region with ones
     self.labels.append(mask) 
         
