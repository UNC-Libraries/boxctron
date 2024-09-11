import csv
import os
from pathlib import Path
from src.utils.json_utils import to_json
from src.utils.bounding_box_utils import draw_bounding_boxes, get_box_coords
from src.utils.common_utils import log
from src.utils.image_normalizer import ImageNormalizer
from src.utils.common_utils import rebase_path
from PIL import Image
import shutil

class SegmentationReportService:
  def __init__(self, csv_path, report_path, config):
    self.csv_path = csv_path
    self.report_path = report_path
    self.images_path = report_path / 'images'
    self.config = config
    self.original_data = False
    self.output_data = []
    self.normalizer = ImageNormalizer(config)

  def generate(self):
    # create output directory and images subdirectory
    self.images_path.mkdir(parents=True, exist_ok=True)
    # copy in csv file
    shutil.copyfile(self.csv_path, self.report_path / 'data.csv')
    # begin processing csv file
    with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
      datareader = csv.reader(f)
      # skip the headers
      next(datareader, None)
      for row in datareader:
        log(f'Processing: {row[0]}')
        # generate annotated image, write to images directory
        image_path = self.generate_annotated_image(row)
        # convert csv data to output structure
        self.output_data.append(self.csv_to_data(row, image_path))
    # write json data file
    self.write_output_data()

  def generate_annotated_image(self, row):
    normalized_path = self.normalizer.build_output_path(row[0])
    destination_path = rebase_path(self.images_path, Path(row[0]), '.jpg')
    # Create parent directories for destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = []
    coords = get_box_coords(row)
    if coords != None:
      boxes.append(coords)
    # If the bounding box needed to be extended, then draw in the extended version of the box
    extended_box = get_box_coords(row, index = 4)
    if extended_box != None:
      log(f'{row[0]} has an extended bounding box')
      boxes.append(extended_box)

    draw_bounding_boxes(str(normalized_path), str(destination_path), [800, 800], boxes, retain_ratio = True)
    return destination_path

  def csv_to_data(self, row, image_path):
    rel_path = image_path.relative_to(self.report_path)
    boxes = get_box_coords(row)
    return {
      'original' : row[0],
      'pred_class' : row[1],
      'pred_conf' : row[2],
      'problem' : bool(row[4]),
      'image' : str(rel_path)
    }

  def write_output_data(self):
    data_wrapper = { 'data' : self.output_data }
    to_json(data_wrapper, self.report_path / 'data.json')
    