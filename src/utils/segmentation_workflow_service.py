import csv
import os
import traceback
from pathlib import Path
from src.utils.image_segmenter import ImageSegmenter
from src.utils.image_normalizer import ImageNormalizer
from src.utils.progress_tracker import ProgressTracker
from src.utils.segmentation_utils import round_box_to_edge, pixels_to_norms, norms_to_pixels
from src.utils.bounding_box_utils import is_problematic_box, extend_bounding_box_to_edges, InvalidBoundingBoxException
import torch
from PIL import Image

# Service which accepts a list of image original files, then performs object
# detection/segmentation to make predictions on normalized versions of those images.
# The outcome is written to a CSV file at the provided report_path.
class SegmentationWorkflowService:
  CSV_HEADERS = ['original_path', 'normalized_path', 'predicted_class', 'predicted_conf', 'bounding_box', 'extended_box']

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
          path = path.resolve()
          normalized_path = self.normalizer.process(path)
          top_predicted, top_score = self.segmenter.predict(normalized_path)
          box_coords = top_predicted['boxes']
          box_norms = None
          extended_box = None
          predicted_class = 0
          orig_box, norm_box = None, None
          # If a bounding box was returned, then convert coordinates to percentages and round to edges
          if box_coords.shape[0] == 1:
            predicted_class = 1
            box_coords = box_coords[0].detach().numpy()
            box_norms = self.normalize_coords(box_coords)
            # Round the bounding box to the edges of the image if they are close
            box_norms = list(round_box_to_edge(box_norms))
            # If box isn't usable for cropping, try extending to edges
            if is_problematic_box(box_norms):
              try:
                extended_box = extend_bounding_box_to_edges(box_norms)
                print(f"   Problem detected with bounding box, extending to edges.")
              except InvalidBoundingBoxException as e:
                print(e.message)
          csv_writer.writerow([path, normalized_path, predicted_class, "{:.4f}".format(top_score), box_norms, extended_box])
          self.progress_tracker.record_completed(path)
        except (KeyboardInterrupt, SystemExit) as e:
          exit(1)
        except BaseException as e:
          print(f'Failed to process {path}: {e}')
          print(traceback.format_exc())

  def normalize_coords(self, box_coords):
    return pixels_to_norms(box_coords, self.config.max_dimension, self.config.max_dimension)
