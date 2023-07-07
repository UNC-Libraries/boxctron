from pathlib import Path
import json
import importlib

class ClassifierConfig:
  def __init__(self, path):
    with open(path) as json_data:
      data = json.load(json_data)
      # Width of the hidden layer used in between the foundation model and the file result
      self.model_width = data.get('model_width', 256)
      # Only predictions with confidence higher than this threshold will be counted as matching the class
      self.predict_rounding_threshold = data.get('predict_rounding_threshold', 0.7)
      # Max dimension size which images will be normalized to
      self.max_dimension = data.get('max_dimension', None)
      self.min_dimension = data.get('min_dimension', None)
      self.output_base_path = Path(data['output_base_path']) if 'output_base_path' in data else None
      self.src_base_path = Path(data['src_base_path']) if 'src_base_path' in data else None
      self.force = bool(data.get('force', False))
