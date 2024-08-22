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
  BATCH_SIZE = 10

  def __init__(self, config, report_path, restart = False):
    self.config = config
    self.report_path = report_path
    self.normalizer = ImageNormalizer(config)
    self.segmenter = ImageSegmenter(config)
    progress_log_path = report_path.parent / (report_path.stem + "_progress.log")
    self.progress_tracker = ProgressTracker(progress_log_path)
    if restart:
      print(f'Restarting progress tracking and reporting')
      self.progress_tracker.reset_log()
      self.report_path.unlink()

  def process(self, paths):
    total = len(paths)
    is_new_file = not Path.exists(self.report_path) or os.path.getsize(self.report_path) == 0
    batch_size = int(self.config.batch_size)

    with open(self.report_path, "a", newline="") as csv_file:
      csv_writer = csv.writer(csv_file)
      # Add headers to file if it is empty
      if is_new_file:
        csv_writer.writerow(self.CSV_HEADERS)

      batch_orig_paths = []
      batch_norm_paths = []
      for idx, path in enumerate(paths):
        path = path.resolve()
        if self.progress_tracker.is_complete(path):
          print(f"Skipping {idx + 1} of {total}: {path}")
          continue
        if self.unprocessable_filename(path):
          print(f"Skipping {idx + 1} of {total} due to filename: {path}")
          self.progress_tracker.record_completed(path)
          continue

        print(f"Processing {idx + 1} of {total}: {path} {len(batch_norm_paths)} {len(batch_orig_paths)} / {batch_size}")
        path = path.resolve()
        try:
          batch_norm_paths.append(self.normalizer.process(path))
          batch_orig_paths.append(path)
        except (KeyboardInterrupt, SystemExit) as e:
          exit(1)
        except BaseException as e:
          print(f'Failed to process {path}: {e}')
          print(traceback.format_exc())

        # Accumulated a batch worth of images, or this is the final image
        if len(batch_orig_paths) >= batch_size or idx == (len(paths) - 1):
          top_predictions, top_scores = self.segmenter.predict(batch_norm_paths)
          for batch_idx, orig_path in enumerate(batch_orig_paths):
            normalized_path = batch_norm_paths[batch_idx]
            top_predicted = top_predictions[batch_idx]
            top_score = top_scores[batch_idx]
            try:
              box_coords = top_predicted['boxes']
              box_norms = None
              extended_box = None
              predicted_class = 0
              orig_box, norm_box = None, None
              # If a bounding box was returned, then convert coordinates to percentages and round to edges
              if box_coords.shape[0] == 1:
                predicted_class = 1
                box_coords = box_coords[0].detach().cpu().numpy()
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
                    # Set the predicted class to 2, to indicate its an invalid prediction
                    predicted_class = 2
              csv_writer.writerow([self.reported_original_path(orig_path),
                                   normalized_path,
                                   predicted_class,
                                   "{:.4f}".format(top_score),
                                   box_norms,
                                   extended_box])
              self.progress_tracker.record_completed(orig_path)
            except (KeyboardInterrupt, SystemExit) as e:
              exit(1)
            except BaseException as e:
              print(f'Failed to process {orig_path}: {e}')
              print(traceback.format_exc())
          batch_orig_paths = []
          batch_norm_paths = []

  # The original path in the form that should be included in the data report
  def reported_original_path(self, orig_path):
    if self.config.remove_src_path_base:
      return Path(str(orig_path).removeprefix(str(self.config.src_base_path)))
    else:
      return orig_path

  def normalize_coords(self, box_coords):
    return pixels_to_norms(box_coords, self.config.max_dimension, self.config.max_dimension)

  def unprocessable_filename(self, path):
    # ignore sidecar files
    return path.stem.startswith("._")
