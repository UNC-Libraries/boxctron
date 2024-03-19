from PIL import Image, ImageDraw
import itertools
import json
from pathlib import Path
from src.utils.segmentation_utils import pixels_to_norms, norms_to_pixels

_all__ = ["_draw_bounding_boxes", "_draw_result_bounding_boxes", "_is_problematic_box", "_number_sides_at_image_edge", "_get_box_coords", "_extend_bounding_box_to_edges"]

def draw_bounding_boxes(img_path, output_path, resize_dims, boxes, retain_ratio = False):
  """
  Draws a list of bounding boxes on an image in different colors, then saves them to file

  Args:
      img_path (str): Path to the input image file.
      output_path (str): Path to the output image file
      resize_dims (list): dimensions to resize image to, in form [w, h]
      boxes (list): List of normalized coordinates (range 0-1) in the format [[x1, y1, x2, y2]], with shape (N, 4).
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

    colors = itertools.cycle(["red", "green", "blue", "yellow"])
    for i, bounding_box in enumerate(boxes):
      color = next(colors)
      if bounding_box:
        box_coords = norms_to_pixels(bounding_box, resized_w, resized_h)
        draw.rectangle(box_coords, outline=color, width=4)

    img.save(output_path)

def draw_result_bounding_boxes(img_paths, base_output_path, resize_dims, target_boxes, predicted_boxes):
  for entry in zip(img_paths, target_boxes, predicted_boxes):
    img_path = Path(entry[0])
    draw_bounding_boxes(img_path, base_output_path / img_path.name, resize_dims, [entry[1], entry[2]])

# Box is problematic if 3 of its sides don't touch the edges of the image
def is_problematic_box(coords):
  if coords == None:
    return False
  count = number_sides_at_image_edge(coords)
  return count != 3

def number_sides_at_image_edge(coords):
  count = 0
  count += coords[0] == 0
  count += coords[1] == 0
  count += coords[2] == 1
  count += coords[3] == 1
  return count

# Load the coordinates of a bounding box from a CSV row
def get_box_coords(row, index = 4):
  if row[index]:
    box_coords = json.loads(row[index])
    return box_coords
  return None

# Used to extend a bounding box that is only touching 1 or 2 image edges, so that it touches 3
# edges so that it is usable for cropping
def extend_bounding_box_to_edges(box_coords):
  coords = box_coords.copy()
  horizontal_length = coords[2] - coords[0]
  vertical_length = coords[3] - coords[1]
  # don't extend if it'll produce a bounding box greater than or equal to half the image
  if vertical_length >= 0.5 and horizontal_length >= 0.5:
    raise InvalidBoundingBoxException("Cannot extend bounding box to image edges, total size of bounding box is too large")
  if vertical_length == horizontal_length:
    raise InvalidBoundingBoxException("Cannot extend bounding box to image edges, sides are equal length")
  left_edge = coords[0] == 0
  right_edge = coords[2] == 1
  top_edge = coords[1] == 0
  bottom_edge = coords[3] == 1
  if (left_edge or right_edge) and (bottom_edge or top_edge):
    # bounding box touches two edges, so extend longest edge
    if vertical_length > horizontal_length:
      coords[1] = 0.0
      coords[3] = 1.0
    else:
      coords[0] = 0.0
      coords[2] = 1.0
  else:
    # Only one side on edge, so extend the two sides perpendicular to that side to the edges
    if left_edge or right_edge:
      coords[1] = 0.0
      coords[3] = 1.0
    else:
      coords[0] = 0.0
      coords[2] = 1.0
  return coords

class InvalidBoundingBoxException(Exception):
    pass
