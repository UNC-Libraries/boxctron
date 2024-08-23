import csv
import os
from pathlib import Path
from src.utils.json_utils import to_json
from src.utils.bounding_box_utils import draw_bounding_boxes, get_box_coords
from src.utils.common_utils import log
from PIL import Image
import shutil

class SegmentationReportService:
  def __init__(self, csv_path, output_path, norms_relative_path = '/', src_base_path = None):
    self.csv_path = csv_path
    self.output_path = output_path
    self.images_path = output_path / 'images'
    self.norms_relative_path = Path(norms_relative_path).resolve()
    self.src_base_path = None
    if src_base_path != None:
      self.src_base_path = str(Path(src_base_path).resolve())
    self.original_data = False
    self.output_data = []

  def generate(self):
    # create output directory and images subdirectory
    self.images_path.mkdir(parents=True, exist_ok=True)
    # copy html page
    shutil.copyfile('src/reports/seg_report.html', self.output_path / 'report.html')
    # Copy in the data.csv file and adjust paths if necessary
    report_csv = self.copy_data_csv()
    # begin processing csv file
    with open(report_csv, 'r', encoding='utf-8-sig') as f:
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

  # Copies the provided data.csv into the report directory and adjusts source paths if needed
  def copy_data_csv(self):
    dest_path = self.output_path / 'data.csv'
    with open(self.csv_path, 'r', encoding='utf-8-sig') as src_f:
      datareader = csv.reader(src_f)
      with open(dest_path, "a", newline="") as dest_f:
        csv_writer = csv.writer(dest_f)
        headers = next(datareader, None)
        csv_writer.writerow(headers)
        for row in datareader:
          row[0] = self.get_original_path(row[0])
          csv_writer.writerow(row)
    return dest_path

  def generate_annotated_image(self, row):
    normalized_path = Path(row[1]).resolve()
    norm_rel_path = str(normalized_path.relative_to(self.norms_relative_path))
    norm_rel_path = self.get_original_path(norm_rel_path)
    norm_rel_path = norm_rel_path.removeprefix('/')

    destination_path = self.images_path / (str(norm_rel_path) + '.jpg')
    # Create parent directories for destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = []
    coords = get_box_coords(row)
    if coords != None:
      boxes.append(coords)
    # If the bounding box needed to be extended, then draw in the extended version of the box
    extended_box = get_box_coords(row, index = 5)
    if extended_box != None:
      log(f'{row[0]} has an extended bounding box')
      boxes.append(extended_box)

    draw_bounding_boxes(str(normalized_path), str(destination_path), [800, 800], boxes, retain_ratio = True)
    return destination_path

  def csv_to_data(self, row, image_path):
    rel_path = image_path.relative_to(self.output_path)
    boxes = get_box_coords(row)

    return {
      'original' : self.get_original_path(row[0]),
      'pred_class' : row[2],
      'pred_conf' : row[3],
      'problem' : bool(row[5]),
      'image' : str(rel_path)
    }

  def get_original_path(self, orig_path):
    if self.src_base_path != None:
      return str(Path(orig_path).resolve()).removeprefix(self.src_base_path)
    return orig_path

  def write_output_data(self):
    data_wrapper = { 'data' : self.output_data }
    to_json(data_wrapper, self.output_path / 'data.json')
    