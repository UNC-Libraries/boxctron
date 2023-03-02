from pathlib import Path
import json

class Config:
  # output_base_path - base path of destination dir, output will be created relative to this
  # src_base_path - base path of source files, paths will be calculated relative to this
  # max_dimension - max size of longest side for images, in pixels. The longest side will be reduced to this value, and the other side scaled accordingly
  def __init__(self, path=None):
    self.max_dimension = None
    self.min_dimension = None
    self.output_base_path = Path('.')
    self.src_base_path = None
    self.force = False
    if path != None:
      self.load_config(path)


  def load_config(self, path):
    with open(path) as json_data:
      data = json.load(json_data)
      self.max_dimension = data.get('max_dimension', None)
      self.min_dimension = data.get('min_dimension', None)
      self.output_base_path = Path(data['output_base_path']) if 'output_base_path' in data else None
      self.src_base_path = Path(data['src_base_path']) if 'src_base_path' in data else None
      self.force = bool(data.get('force', False))