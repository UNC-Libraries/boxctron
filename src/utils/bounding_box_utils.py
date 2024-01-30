from PIL import Image, ImageDraw
import itertools
from pathlib import Path

_all__ = ["_evaluate_iou", "_evaluate_giou"]

def draw_bounding_boxes(img_path, output_path, resize_dims, boxes):
  """
  Draws a list of bounding boxes on an image in different colors, then saves them to file

  Args:
      img_path (str): Path to the input image file.
      output_path (str): Path to the output image file
      resize_dims (list): dimensions to resize image to, in form [w, h]
      boxes (list): List of tensors containing coordinates in the format [[x1, y1, x2, y2]], with shape (N, 4).
  """
  with Image.open(img_path) as img:
    img = img.resize(resize_dims)
    draw = ImageDraw.Draw(img)

    colors = itertools.cycle(["green", "red", "blue", "yellow"])
    for i, bounding_box in enumerate(boxes):
      color = next(colors)
      box_list = bounding_box.tolist()
      if box_list:
        draw.rectangle(box_list, outline=color, width=4)

    img.save(output_path)

def draw_result_bounding_boxes(img_paths, base_output_path, resize_dims, target_boxes, predicted_boxes):
  for entry in zip(img_paths, target_boxes, predicted_boxes):
    img_path = Path(entry[0])
    draw_bounding_boxes(img_path, base_output_path / img_path.name, resize_dims, [entry[1], entry[2]])