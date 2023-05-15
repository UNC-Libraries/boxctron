from pathlib import Path
import json

class AugmentConfig:
  # output_base_path - base path of destination dir, output will be created relative to this
  # src_base_path - base path of source files, paths will be calculated relative to this
  def __init__(self, path=None):
    self.output_base_path = Path('.')
    self.base_image_path = Path('/')
    self.annotations_path = None
    self.annotations_output_path = None
    self.file_list_path = None
    self.file_list_output_path = None
    if path != None:
      self.load_config(path)

  def load_config(self, path):
    with open(path) as json_data:
      data = json.load(json_data)
      self.output_base_path = Path(data['output_base_path']) if 'output_base_path' in data else None
      # Base path where image paths should be evaluated against
      self.base_image_path = Path(data.get('base_image_path', '/')).resolve()
      self.annotations_path = Path(data['annotations_path']) if 'annotations_path' in data else None
      self.annotations_output_path = Path(data['annotations_output_path']) if 'annotations_output_path' in data else None
      self.file_list_path = Path(data['file_list_path']) if 'file_list_path' in data else None
      self.file_list_output_path = Path(data['file_list_output_path']) if 'file_list_output_path' in data else None