import csv
import os
import traceback
from pathlib import Path
from src.utils.image_segmenter import ImageSegmenter
from src.utils.image_normalizer import ImageNormalizer
from src.utils.progress_tracker import ProgressTracker
import torch
from PIL import Image

# Service which accepts a list of image original files, then performs object
# detection/segmentation to make predictions on normalized versions of those images.
# The outcome is written to a CSV file at the provided report_path.
class SegmentationWorkflowService:
  CSV_HEADERS = ['original_path', 'normalized_path', 'predicted_class', 'predicted_conf', 'orig_box', 'norm_box']

  def __init__(self, config, report_path, restart = False):
    self.config = config
    self.report_path = report_path
    self.normalizer = ImageNormalizer(config)
    self.segmenter = ImageSegmenter(config)
    self.progress_tracker = ProgressTracker(config.progress_log_path)
    if restart:
      print(f'Restarting progress tracking and reporting')
      self.progress_tracker.reset_log()
      self.report_path.unlink()

  def process(self, paths):
    total = len(paths)
    is_new_file = not Path.exists(self.report_path) or os.path.getsize(self.report_path) == 0

    with open(self.report_path, "a", newline="") as csv_file:
      csv_writer = csv.writer(csv_file)
      # Add headers to file if it is empty
      if is_new_file:
        csv_writer.writerow(self.CSV_HEADERS)

      for idx, path in enumerate(paths):
        if self.progress_tracker.is_complete(path):
          print(f"Skipping {idx + 1} of {total}: {path}")
          continue

        print(f"Processing {idx + 1} of {total}: {path}")
        try:
          normalized_path = self.normalizer.process(path)
          top_predicted, top_score = self.segmenter.predict(normalized_path)
          box_coords = top_predicted['boxes']
          predicted_class = 0
          orig_box, norm_box = None, None
          if box_coords.shape[0] == 1:
            predicted_class = 1
            box_coords = box_coords[0].detach().numpy()
            orig_box = self.rescale_bounding_box(box_coords, path)
            norm_box = self.rescale_bounding_box(box_coords, normalized_path)

          csv_writer.writerow([path, normalized_path, predicted_class, "{:.4f}".format(top_score), orig_box, norm_box])
          self.progress_tracker.record_completed(path)
        except (KeyboardInterrupt, SystemExit) as e:
          exit(1)
        except BaseException as e:
          print(f'Failed to process {path}: {e}')
          print(traceback.format_exc())

  def rescale_bounding_box(self, bounding_box, src_image_path):
    w, h = None, None
    with Image.open(src_image_path) as img:
      w, h = img.width, img.height
    # Dimension images were scales to in order to process them with the model
    max_dim = self.config.max_dimension
    x1, y1 = bounding_box[0] / max_dim * w, bounding_box[1] / max_dim * h
    x2, y2 = bounding_box[2] / max_dim * w, bounding_box[3] / max_dim * h
    return [x1, y1, x2, y2]

