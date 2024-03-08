import csv
import os
from pathlib import Path
from src.utils.json_utils import to_json
from src.utils.bounding_box_utils import draw_bounding_boxes
from src.utils.common_utils import log
import json
import shutil

class SegmentationReportService:
  def __init__(self, csv_path, output_path, norms_relative_path = '/'):
    self.csv_path = csv_path
    self.output_path = output_path
    self.images_path = output_path / 'images'
    self.norms_relative_path = Path(norms_relative_path).resolve()
    self.original_data = False
    self.output_data = []

  def generate(self):
    # create output directory and images subdirectory
    self.images_path.mkdir(parents=True, exist_ok=True)
    # copy html page
    shutil.copyfile('src/reports/seg_report.html', self.output_path / 'report.html')
    # copy in csv file
    shutil.copyfile(self.csv_path, self.output_path / 'data.csv')
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
    normalized_path = Path(row[1]).resolve()
    norm_rel_path = normalized_path.relative_to(self.norms_relative_path)
    destination_path = self.images_path / (str(norm_rel_path) + '.jpg')
    # Create parent directories for destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = []
    if row[5]:
      boxes.append(json.loads(row[5]))
    draw_bounding_boxes(str(normalized_path), str(destination_path), [800, 800], boxes, retain_ratio = True)
    return destination_path

  def csv_to_data(self, row, image_path):
    rel_path = image_path.relative_to(self.output_path)
    return {
      'original' : row[0],
      'pred_class' : row[2],
      'pred_conf' : row[3],
      'image' : str(rel_path)
    }

  def write_output_data(self):
    data_wrapper = { 'data' : self.output_data }
    to_json(data_wrapper, self.output_path / 'data.json')
    