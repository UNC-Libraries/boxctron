from numpy import zeros
from src.datasets.color_bar_dataset import ColorBarDataset
from src.utils.resnet_utils import load_for_resnet, load_mask_for_resnet
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

from PIL import Image

# Dataset class for segmenting images into color bar and subject (negative) regions
# Returns image, target tuples as two tensors:
# A normalized image of (3, h, w) and a mask of (1, h, w)
# Where h and w are image dimensions after being resized for resnet
class ColorBarSegmentationDataset(ColorBarDataset):
  def __init__(self, config, image_paths, split = 'train'):
    super().__init__(config, image_paths, split)
  
  # Must be overriden from parent class
  def __getitem__(self, index):
    image_data = load_for_resnet(self.image_paths[index], self.config.max_dimension)
    label_mask = load_mask_for_resnet(self.labels[index], self.config.max_dimension)
    label_mask = label_mask.bool()
    return image_data, label_mask

  # Helper function for displaying masks imposed on transformed image tensors
  def visualize_tensor(self, img, mask):
    img = (img.clamp(0, 1) * 255).byte() # Scales back to uint8 for compatibility
    masking = draw_segmentation_masks(img, mask, alpha=0.7, colors="blue")
    masking = to_pil_image(masking)
    masking.show()

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
      image_labels = path_to_labels[str(image_path)]
      mask = zeros((h, w), dtype=bool)
      for label in image_labels:
        if 'color_bar' in label['rectanglelabels']:
          width, height = label['width'], label['height']
          x = int(label['x']/100 * label['original_width'])
          y = int(label['y']/100 * label['original_height'])
          x2 = x + int(width/100 * label['original_width']) # bar width
          y2 = y + int(height/100 * label['original_height']) # bar height
          mask[y:y2, x:x2] = 1 # Mark all pixels in the masked region with ones
      self.labels.append(mask) 