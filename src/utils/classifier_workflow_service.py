import csv
import os
from pathlib import Path
from src.utils.image_classifier import ImageClassifier
from src.utils.image_normalizer import ImageNormalizer

# Service which accepts a list of image original files, then uses normalized versions of 
# those files to use a classifier to make predictions about those images.
# The outcome is written to a CSV file at the provided report_path.
class ClassifierWorkflowService:
  CSV_HEADERS = ['original_path', 'normalized_path', 'predicted_class', 'predicted_conf']

  def __init__(self, config, report_path):
    self.config = config
    self.report_path = report_path
    self.normalizer = ImageNormalizer(config)
    self.classifier = ImageClassifier(config)

  def process(self, paths):
    total = len(paths)
    is_new_file = not Path.exists(self.report_path) or os.path.getsize(self.report_path) == 0

    with open(self.report_path, "a", newline="") as csv_file:
      csv_writer = csv.writer(csv_file)
      # Add headers to file if it is empty
      if is_new_file:
        csv_writer.writerow(self.CSV_HEADERS)

      for idx, path in enumerate(paths):
        print(f"Processing {idx + 1} of {total}: {path}")
        try:
          normalized_path = self.normalizer.process(path)
          results = self.classifier.predict(normalized_path)
          csv_writer.writerow([path, normalized_path, results[1][0].item(), "{:.4f}".format(results[0][0].item())])
        except (KeyboardInterrupt, SystemExit) as e:
          exit(1)
        except BaseException as e:
          print(f'Failed to process {path}: {e}')
