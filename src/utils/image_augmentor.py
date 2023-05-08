from PIL import Image, ImageEnhance
from pathlib import Path
import random
import logging

# Utility for normalizing images based on a configuration.
# Currently normalizes all files to JPGs
class ImageAugmentor:
  def __init__(self, config):
    self.config = config

  # Augment an image to the expected configuration, saving the new versions to an configured output path
  def process(self, path):
    with Image.open(path) as img:
      img, rotate_type = self.aug_rotation(img)
      img, satur_type = self.aug_saturation(img)

      output_path = self.build_output_path(path, [rotate_type, satur_type])
      print(f'Output path {output_path}')
      # construct path to write to, then save the file
      output_path.parent.mkdir(exist_ok=True)
      img.save(output_path, "JPEG", optimize=True, quality=80)
      return output_path

  def build_output_path(self, path, aug_types):
    print(f'Base: {self.config.src_base_path}')
    if self.config.src_base_path == None:
      rel_path = path.name
    else:
      rel_path = str(path.relative_to(self.config.src_base_path))
    filename = path.stem
    dest = self.config.output_base_path / rel_path
    print(f'Dest: {dest}')
    # add augmentations to filename
    return dest.with_stem(f'{filename}_{"_".join(aug_types)}')

  def aug_rotation(self, img):
    index = random.randrange(0, 5)
    print(f'Indexr {index}')
    match index:
      case 0:
        return img.rotate(90, expand=1), 'r90'
      case 1:
        return img.transpose(Image.FLIP_TOP_BOTTOM), 'rfv'
      case 2:
        return img.rotate(90, expand=1).transpose(Image.FLIP_LEFT_RIGHT), 'r90fh'
      case 3:
        return img.transpose(Image.FLIP_LEFT_RIGHT), 'rfh'
      case 4:
        return img.rotate(random.randrange(1, 10)), 'rsmall'

  def aug_saturation(self, img):
    index = random.randrange(0, 10)
    print(f'Indexs {index}')
    if index == 0:
      converter = ImageEnhance.Color(img)
      return converter.enhance(0.75), 's75'
    elif index == 1:
      converter = ImageEnhance.Color(img)
      return converter.enhance(1.25), 's125'
    else:
      return img, 's100'