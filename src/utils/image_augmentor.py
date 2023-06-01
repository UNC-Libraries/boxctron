from PIL import Image, ImageEnhance
from pathlib import Path
import random
import logging
from src.utils.json_utils import from_json, to_json
import copy
import shutil

# Utility for normalizing images based on a configuration.
# Currently normalizes all files to JPGs
class ImageAugmentor:
  def __init__(self, config):
    self.config = config
    self.load_annotations()
    self.init_file_list()

  # Augment an image to the expected configuration, saving the new versions to an configured output path
  def process(self, path):
    with Image.open(path) as img:
      img, rotate_type = self.aug_rotation(img)
      img, satur_type = self.aug_saturation(img)

      output_path = self.build_output_path(path, [rotate_type, satur_type])
      # skip saving file and adding annotation if one exists with the same name
      if str(output_path.resolve()) in self.path_to_anno:
        return output_path
      # construct path to write to, then save the file
      output_path.parent.mkdir(parents=True, exist_ok=True)
      img.save(output_path, "JPEG", optimize=True, quality=80)

      self.add_aug_annotation(path, output_path)
      self.add_to_file_list(output_path)
      return output_path

  def add_aug_annotation(self, orig_path, output_path):
    orig_anno = self.path_to_anno[str(orig_path.resolve())]
    aug_anno = copy.deepcopy(orig_anno)
    aug_anno['image'] = str(output_path)
    self.annotations.append(aug_anno)

  def init_file_list(self):
    if self.config.file_list_path.exists() and self.config.file_list_path != self.config.file_list_output_path:
      shutil.copy(self.config.file_list_path, self.config.file_list_output_path)

  def add_to_file_list(self, output_path):
    with open(self.config.file_list_output_path, 'a') as file_list:
      file_list.write(f'\n{output_path}')

  # Load the original annotations from file, and build a lookup map for filepaths
  def load_annotations(self):
    self.annotations = from_json(self.config.annotations_path)
    self.path_to_anno = {}
    for anno in self.annotations:
      # Skip over images already in 
      if anno['image'].startswith('http://localhost'):
        img_path = (self.config.base_image_path / anno['image'].split('/', 3)[3]).resolve()
      else:
        img_path = anno['image']
      self.path_to_anno[str(img_path)] = anno

  # Write annotations out to their output path
  def persist_annotations(self):
    to_json(self.annotations, self.config.annotations_output_path)

  def build_output_path(self, path, aug_types):
    rel_path = str(path.relative_to(self.config.base_image_path))
    filename = path.stem
    dest = self.config.output_base_path / rel_path
    # add augmentations to filename
    return dest.with_name(f'{filename}_{"_".join(aug_types)}{path.suffix}')

  def aug_rotation(self, img):
    index = random.randrange(0, 5)
    if index == 0:
      return img.rotate(90, expand=1), 'r90'
    elif index == 1:
      return img.transpose(Image.FLIP_TOP_BOTTOM), 'rfv'
    elif index == 2:
      return img.rotate(90, expand=1).transpose(Image.FLIP_LEFT_RIGHT), 'r90fh'
    elif index == 3:
      return img.transpose(Image.FLIP_LEFT_RIGHT), 'rfh'
    else:
      return img.rotate(random.randrange(1, 10)), 'rsmall'

  def aug_saturation(self, img):
    index = random.randrange(0, 10)
    if index == 0:
      converter = ImageEnhance.Color(img)
      return converter.enhance(0.75), 's75'
    elif index == 1:
      converter = ImageEnhance.Color(img)
      return converter.enhance(1.25), 's125'
    else:
      return img, 's100'
