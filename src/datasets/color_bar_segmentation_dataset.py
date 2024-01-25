from numpy import zeros
from src.datasets.color_bar_dataset import ColorBarDataset
from src.utils.resnet_utils import load_for_resnet, load_mask_for_resnet
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import box_area
import torch
from torchvision import transforms

from PIL import Image

# Dataset class for segmenting images into color bar and subject (negative) regions
# Returns image, target tuples as two tensors:
# A normalized image of (3, h, w) and a mask of (1, h, w)
# Where h and w are image dimensions after being resized for resnet
class ColorBarSegmentationDataset(ColorBarDataset):
  ROUNDING_THRESHOLD = 2.5

  def collate_zip_fn(data):
    zipped = zip(*data)
    return tuple(zipped)

  collate_fn = collate_zip_fn

  def __init__(self, config, image_paths, split = 'train'):
    self.boxes = []
    self.masks = []
    super().__init__(config, image_paths, split)
  
  # Must be overriden from parent class
  def __getitem__(self, index):
    input_image = Image.open(self.image_paths[index])
    preprocess = transforms.Compose([
        # Resize image to standard dimensions, no padding
        transforms.Resize((self.config.max_dimension, self.config.max_dimension)),
        transforms.ToTensor(),
    ])
    image_data = preprocess(input_image)
    # Convert to a pytorchvision image
    image_data = tv_tensors.Image(image_data)
    target = {}
    # label_mask = load_mask_for_resnet(self.masks[index], self.config.max_dimension)
    # label_mask = label_mask.bool()
    target = {
      'boxes' : tv_tensors.BoundingBoxes(self.boxes[index], format="XYXY", canvas_size=F.get_size(image_data)),
      'area' : box_area(self.boxes[index]),
      'image_id' : torch.tensor([index], dtype=torch.int64),
      # 'masks' : [label_mask],
      'labels' : torch.tensor(self.labels[index], dtype=torch.int64)
      # 'img_path' : str(self.image_paths[index])
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
      w, h = None, None
      with Image.open(image_path) as img:
        w, h = img.width, img.height
      self.image_dimensions.append((w, h))
      image_labels = path_to_labels[str(image_path)]
      # mask = zeros((h, w), dtype=bool)
      bounding_boxes = []
      bar_box = None
      labels = [0]
      original_width, original_height = w, h

      for label in image_labels:
        if 'color_bar' in label['rectanglelabels']:
          norm_x, norm_y = self.round_to_edge(label['x']), self.round_to_edge(label['y'])
          norm_x2, norm_y2 = self.round_to_edge(label['x'] + w), self.round_to_edge(label['y'] + h)
          x1, y1, x2, y2 = self.norms_to_pixels(norm_x, norm_y, norm_x2, norm_y2, w, h)
          bgnx1, bgny1, bgnx2, bgny2 = self.background_box((norm_x, norm_y, norm_x2, norm_y2))
          bgx1, bgy1, bgx2, bgy2 = self.norms_to_pixels(bgnx1, bgny1, bgnx2, bgny2, w, h)
          bounding_boxes.append([bgx1, bgy1, bgx2, bgy2])
          # mask[y1:y2, x1:x2] = 1 # Mark all pixels in the masked region with ones
          bar_box = [x1, y1, x2, y2]
          bounding_boxes.append(bar_box)
          labels.append(1)
      self.labels.append(labels)
      # self.masks.append(mask)
      if len(bounding_boxes) == 0:
        bounding_boxes.append([0, 0, w, h])
      self.boxes.append(torch.tensor(bounding_boxes, dtype=torch.float32))

  def norms_to_pixels(self, norm_x, norm_y, norm_x2, norm_y2, width, height):
    x1 = int(norm_x * width)
    y1 = int(norm_y * height)
    x2 = int(norm_x2 * width) # bar width
    y2 = int(norm_y2 * height) # bar height
    return (x1, y1, x2, y2)

  def background_box(self, bar_box):
    if bar_box == None:
      return (0, 0, 1, 1)
    x, y, x2, y2 = 0, 0, 1, 1
    if bar_box[0] == 0 and bar_box[2] != 1:
      x = bar_box[2]
    if bar_box[1] == 0 and bar_box[3] != 1:
      y = bar_box[3]
    if bar_box[0] != 0 and bar_box[2] == 1:
      x2 = bar_box[0]
    if bar_box[1] != 0 and bar_box[3] == 1:
      y2 = bar_box[1]
    return (x, y, x2, y2)

  # Rounds a percentage based coordinate to the nearest edge if it is within the
  # threshold, and converts the percentage to 0-1 form.
  def round_to_edge(self, percent):
    if percent > (100.0 - self.ROUNDING_THRESHOLD):
      return 1.0
    if percent < (0 + self.ROUNDING_THRESHOLD):
      return 0.0
    return percent / 100
