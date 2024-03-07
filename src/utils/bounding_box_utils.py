from PIL import Image, ImageDraw
import itertools
from pathlib import Path
from src.utils.segmentation_utils import pixels_to_norms, norms_to_pixels

_all__ = ["_draw_bounding_boxes", "_draw_result_bounding_boxes"]

def draw_bounding_boxes(img_path, output_path, resize_dims, boxes, retain_ratio = False):
  """
  Draws a list of bounding boxes on an image in different colors, then saves them to file

  Args:
      img_path (str): Path to the input image file.
      output_path (str): Path to the output image file
      resize_dims (list): dimensions to resize image to, in form [w, h]
      boxes (list): List of coordinates in the format [[x1, y1, x2, y2]], with shape (N, 4).
      retain_ratio (boolean): if true, resize dimensions will be treated as max dimensions
          rather than exact dimensions in order to retain original aspect ratio.
  """
  with Image.open(img_path) as img:
    start_w, start_h = img.width, img.height
    
    if retain_ratio:
      img = img.copy()
      img.thumbnail(resize_dims)
    else:
      img = img.resize(resize_dims)
    resized_w, resized_h = img.width, img.height
    draw = ImageDraw.Draw(img)

    colors = itertools.cycle(["green", "red", "blue", "yellow"])
    for i, bounding_box in enumerate(boxes):
      color = next(colors)
      if bounding_box:
        norms = pixels_to_norms(bounding_box, start_w, start_h)
        box_coords = norms_to_pixels(norms, resized_w, resized_h)
        print(f'Start size {start_w} x {start_h}')
        print(f'Resize size {resized_w} x {resized_h}')
        print(f'Sart coords\n{bounding_box}\n{box_coords}')
        draw.rectangle(box_coords, outline=color, width=4)

    img.save(output_path)

def draw_result_bounding_boxes(img_paths, base_output_path, resize_dims, target_boxes, predicted_boxes):
  for entry in zip(img_paths, target_boxes, predicted_boxes):
    img_path = Path(entry[0])
    draw_bounding_boxes(img_path, base_output_path / img_path.name, resize_dims, [entry[1], entry[2]])