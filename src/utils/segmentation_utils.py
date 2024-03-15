import torch

_all__ = ["_get_top_predicted", "_get_top_scores", "_background_box", "_round_box_to_edge", "_norms_to_pixels", "_pixels_to_norms"]

ROUNDING_THRESHOLD = 0.025

# Takes output from the model for one item, and selects the bounding box with
# the highest score, assuming its higher than the minimum score threshold
def get_top_predicted(predict_rounding_threshold, out_entry):
  scores = out_entry['scores']
  if not torch.any(scores > predict_rounding_threshold):
    return {
    'boxes' : torch.zeros((0, 4), dtype=torch.float32),
    'labels' : torch.tensor([]),
    'scores' : torch.zeros((0, 4), dtype=torch.float32)
  }

  top_index = scores.argmax().item()
  return {
    'boxes' : out_entry['boxes'][top_index].unsqueeze(0),
    'labels' : torch.tensor([1]),
    'scores' : out_entry['scores'][top_index].unsqueeze(0)
  }

# Given the outputs from a segmentation model in eval mode, find the highest
# prediction score for each image processed. Return the list of scores.
def get_top_scores(outs):
  top = []
  for entry in outs:
    scores = entry['scores']
    if scores.shape[0] == 0:
      top.append(0)
    else:
      top.append(scores.max().item())
  return top


# Given a bounding box representing a color bar segment, invert that bounding box
# so that it contains all of the image NOT in the provided bounding box
def background_box(bar_box):
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

# Given a bounding box with 0-1 coordinates, round coordinates of the box to the edges of the image 
# if the coordinates are within the coord_round_threshold
# returns rounded bounding box in 0-1 based coordinates.
def round_box_to_edge(bounding_box):
  x, y, x2, y2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
  norm_x, norm_y = round_to_edge(x), round_to_edge(y)
  norm_x2, norm_y2 = round_to_edge(x2), round_to_edge(y2)
  if norm_x == norm_x2:
    norm_x = x
    norm_x2 = x2
  if norm_y == norm_y2:
    norm_y = y
    norm_y2 = y2
  return (norm_x, norm_y, norm_x2, norm_y2)

# Rounds a 0-1 based coordinate to the nearest edge if it is within the threshold
def round_to_edge(coord):
  if coord > (1.0 - ROUNDING_THRESHOLD):
    return 1.0
  if coord < (0 + ROUNDING_THRESHOLD):
    return 0.0
  return coord

def norms_to_pixels(norms, width, height):
  if len(norms) == 0:
    return ()
  x1 = int(norms[0] * width)
  y1 = int(norms[1] * height)
  x2 = int(norms[2] * width) # bar width
  y2 = int(norms[3] * height) # bar height
  return (x1, y1, x2, y2)

def pixels_to_norms(coords, width, height):
  if len(coords) == 0:
    return ()
  norm_x1 = coords[0] / width
  norm_y1 = coords[1] / height
  norm_x2 = coords[2] / width
  norm_y2 = coords[3] / height
  return (norm_x1, norm_y1, norm_x2, norm_y2)