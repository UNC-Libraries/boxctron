from numpy import zeros
from src.datasets.color_bar_dataset import ColorBarDataset
from src.utils.resnet_utils import load_for_resnet, load_mask_for_resnet
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from src.utils.segmentation_utils import round_box_to_edge, norms_to_pixels
from torchvision.ops import box_area
import torch
from torchvision import transforms

from PIL import Image

# Dataset class for segmenting images into color bar and subject (negative) regions
# Returns image, target tuples as two tensors:
# A normalized image of (3, h, w) and a mask of (1, h, w)
# Where h and w are image dimensions after being resized for resnet
class ColorBarSegmentationDataset(ColorBarDataset):
  def collate_zip_fn(data):
    zipped = zip(*data)
    return tuple(zipped)

  collate_fn = collate_zip_fn

  def __init__(self, config, image_paths, split = 'train'):
    self.boxes = []
    self.masks = []
    super().__init__(config, image_paths, split)

  def normalize_image(path, max_dimension):
    input_image = Image.open(path)
    preprocess = transforms.Compose([
        # Resize image to standard dimensions, no padding
        transforms.Resize((max_dimension, max_dimension)),
        transforms.ToTensor(),
    ])
    image_data = preprocess(input_image)
    # Convert to a pytorchvision image
    return tv_tensors.Image(image_data)

  # Must be overriden from parent class
  def __getitem__(self, index):
    image_data = ColorBarSegmentationDataset.normalize_image(self.image_paths[index], self.config.max_dimension)
    target = {}
    target = {
      'boxes' : tv_tensors.BoundingBoxes(self.boxes[index], format="XYXY", canvas_size=F.get_size(image_data)),
      'area' : box_area(self.boxes[index]),
      'image_id' : torch.tensor([index], dtype=torch.int64),
      # 'masks' : [label_mask],
      'labels' : torch.tensor(self.labels[index], dtype=torch.int64),
      'img_path' : str(self.image_paths[index]),
    }
    # print(f'Getitem {target}')
    return image_data, target

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
      w, h = self.config.max_dimension, self.config.max_dimension
      image_labels = path_to_labels[str(image_path)]
      # mask = zeros((h, w), dtype=bool)
      bounding_boxes = []
      bar_box = None
      # labels = [0]
      labels = []
      original_width, original_height = w, h

      for label in image_labels:
        if 'color_bar' in label['rectanglelabels']:
          norms = self.round_label_to_edge(label)
          coords = norms_to_pixels(norms, w, h)
          # bgnx1, bgny1, bgnx2, bgny2 = self.background_box((norm_x, norm_y, norm_x2, norm_y2))
          # bgx1, bgy1, bgx2, bgy2 = self.norms_to_pixels(bgnx1, bgny1, bgnx2, bgny2, w, h)
          # bounding_boxes.append([bgx1, bgy1, bgx2, bgy2])
          # mask[y1:y2, x1:x2] = 1 # Mark all pixels in the masked region with ones
          bar_box = coords
          # bounding_boxes.append(bar_box)
          labels.append(1)
      self.labels.append(labels)
      # self.masks.append(mask)
      if bar_box == None:
        self.boxes.append(torch.zeros((0, 4), dtype=torch.float32))
      else:
        self.boxes.append(torch.tensor([bar_box], dtype=torch.float32))
      # if len(bounding_boxes) == 0:
        # bounding_boxes.append([0, 0, w, h])
      # self.boxes.append(torch.tensor(bounding_boxes, dtype=torch.float32))

  def round_label_to_edge(self, label):
    label_w, label_h = label['width'] / 100, label['height'] / 100
    x, y = label['x'] / 100, label['y'] / 100
    x2, y2 = x + label_w, y + label_h
    return round_box_to_edge([x, y, x2, y2])
