from pathlib import Path
import json
import importlib

class ClassifierConfig:
  def __init__(self, path=None):
    self.predict_rounding_threshold = 0.7
    self.model_path = None
    self.max_dimension = None
    self.min_dimension = None
    self.output_base_path = Path('.')
    self.src_base_path = None
    self.progress_log_path = None
    self.batch_size = None
    self.force = False
    # If true, then the original_path in the output CSV will have the src_base_path trimmed off
    self.remove_src_path_base = False
    if path != None:
      self.load_config(path)

  def load_config(self, path):
    with open(path) as json_data:
      data = json.load(json_data)
      # Only predictions with confidence higher than this threshold will be counted as matching the class
      self.predict_rounding_threshold = float(data.get('predict_rounding_threshold', '0.7'))
      # Classifer model checkpoint
      self.model_path = Path(data['model_path']) if 'model_path' in data else None
      # Max dimension size which images will be normalized to
      self.max_dimension = data.get('max_dimension', None)
      self.min_dimension = data.get('min_dimension', self.max_dimension)
      self.output_base_path = Path(data['output_base_path']).resolve() if 'output_base_path' in data else None
      self.src_base_path = Path(data['src_base_path']).resolve() if 'src_base_path' in data else None
      self.progress_log_path = Path(data['progress_log_path']) if 'progress_log_path' in data else None
      self.batch_size = data.get('batch_size', 10)
      self.force = bool(data.get('force', False))
      self.remove_src_path_base = bool(data.get('remove_src_path_base', False))
