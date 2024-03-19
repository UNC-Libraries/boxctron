import csv
import os
import traceback
from pathlib import Path
from src.utils.image_segmenter import ImageSegmenter
from src.utils.segmentation_utils import round_box_to_edge, pixels_to_norms, norms_to_pixels, background_box
from src.utils.bounding_box_utils import is_problematic_box, get_box_coords, number_sides_at_image_edge
import torch
from PIL import Image
from src.utils.common_utils import log

# Service which accepts a prediction CSV, and generates cropped versions of listed original
# images if they were predicted to contain color bars.
# Cropped images are written to the provided output path.
class CroppingWorkflowService:
  def __init__(self, csv_path, output_path, exclusions_path = None, originals_base_path = Path("/")):
    self.csv_path = csv_path
    self.output_path = output_path
    self.exclusions_path = exclusions_path
    self.originals_base_path = originals_base_path.resolve()
    self.cropped_paths = []
  
  def process(self):
    # Load exclusion paths
    self.load_exclusions()
    # Load segmentation csv report
    with open(self.csv_path, newline='') as csvfile:
      csv_reader = csv.reader(csvfile)
      # Skip the header
      next(csv_reader, None)
      for row in csv_reader:
        original_path = Path(row[0]).resolve()
        if self.is_excluded(original_path) or not self.has_color_bar(row):
          continue
        box_coords = get_box_coords(row)
        if is_problematic_box(box_coords):
          extended_box = get_box_coords(row, index = 5)
          if extended_box == None:
            log(f'ERROR: File {original_path} has problematic bounding box that could not be extrapolated, skipping')
            continue
          else:
            log(f'WARN: File {original_path} has bounding box that has been extrapolated to image edges')
            box_coords = extended_box
        log(f'Cropping {original_path}')
        cropped_path = self.crop_image(original_path, box_coords)
        self.cropped_paths.append(cropped_path)
    return self.cropped_paths

  def has_color_bar(self, row):
    return row[2] == "1"

  def is_excluded(self, path):
    return str(path) in self.exclusions_set

  def load_exclusions(self):
    self.exclusions_set = set()
    if self.exclusions_path is not None:
      with open(self.exclusions_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
          self.exclusions_set.add(row[0])

  def crop_image(self, img_path, box_coords):
    with Image.open(img_path) as img:
      start_w, start_h = img.width, img.height
      # Invert bounding box so that it is the region to retain
      crop_coords = background_box(box_coords)
      # convert crop coords from percentages to pixels
      crop_pixels = norms_to_pixels(crop_coords, start_w, start_h)
      # crop image
      cropped = img.crop(tuple(crop_pixels))
      # write image out to destination path
      if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
      dest_path = self.cropped_image_output_path(img_path)
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      cropped.save(dest_path, "JPEG", optimize=True, quality=80)
      return dest_path

  def cropped_image_output_path(self, img_path):
    relative_path = img_path.relative_to(self.originals_base_path)
    return self.output_path / relative_path
