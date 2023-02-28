from PIL import Image
from pathlib import Path
import logging

# Utility for normalizing images based on a configuration.
# Currently normalizes all files to JPGs
class ImageNormalizer:
  def __init__(self, config):
    self.config = config

  # Normalize an image to the expected configuration, saving the normalized version to an configured output path
  def process(self, path):
    output_path = self.build_output_path(path)
    # Skip regenerating 
    if not self.config.force and output_path.exists():
      logging.info('Derivative already exists, skipping %s', path)
      return

    with Image.open(path) as img:
      if img.mode != "RGB":
        img = img.convert("RGB")
      img = ImageNormalizer.resize_to_max_dimension(img, self.config.max_dimension)
      # construct path to write to, then save the file
      output_path.parent.mkdir(exist_ok=True)
      img.save(output_path, "JPEG", optimize=True, quality=80)

  # Constructs an output path based on the input path and configured base paths.
  def build_output_path(self, path):
    if self.config.src_base_path == None:
      rel_path = path.name
    else:
      rel_path = str(path.relative_to(self.config.src_base_path))
    dest = self.config.output_base_path / rel_path
    # change the extension to jpg
    return dest.with_suffix('.jpg')

  # Resizes the provided image based on the longest dimension, if it exceeds the provided max_dimension.
  # Otherwise, returns the original image.
  def resize_to_max_dimension(img, max_dimension):
    width, height = img.width, img.height
    if max(width, height) <= max_dimension:
      return img
    if width >= height:
      width = max_dimension
      height = int(height * (width / img.width))
    else:
      height = max_dimension
      width = int(width * (height / img.height))
    return img.resize((width, height))