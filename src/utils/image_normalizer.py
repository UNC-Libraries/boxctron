from PIL import Image, ImageFile
from pathlib import Path
import logging
import psutil

# Utility for normalizing images based on a configuration.
# Currently normalizes all files to JPGs
class ImageNormalizer:
  def __init__(self, config):
    # Disable DecompressionBombError since many of our images are huge
    Image.MAX_IMAGE_PIXELS = None
    # Prevent pillow from throwing an exception when reading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    self.config = config

  # Normalize an image to the expected configuration, saving the normalized version to an configured output path
  # Returns the path of the normalized file
  def process(self, path):
    output_path = self.build_output_path(path)
    # Skip regenerating 
    if not self.config.force and output_path.exists():
      logging.info('Derivative already exists, skipping %s', path)
      return output_path


    print(f"Memoryn1: {psutil.virtual_memory()}")
    with Image.open(path) as img:
      print(f"Memoryn2: {psutil.virtual_memory()}")
      if img.mode != "RGB":
        img = img.convert("RGB")
      print(f"Memoryn3: {psutil.virtual_memory()}")
      img = self.resize(img)
      print(f"Memoryn4: {psutil.virtual_memory()}")
      # construct path to write to, then save the file
      output_path.parent.mkdir(parents=True, exist_ok=True)
      img.save(output_path, "JPEG", optimize=True, quality=80)
      print(f"Memoryn5: {psutil.virtual_memory().percent}")
    return output_path

  # Constructs an output path based on the input path and configured base paths.
  def build_output_path(self, path):
    if self.config.src_base_path == None:
      rel_path = path.name
    else:
      rel_path = str(path.relative_to(self.config.src_base_path))
    dest = self.config.output_base_path / rel_path
    # change the extension to jpg
    return dest.with_suffix('.jpg')

  # Resizes the provided image based on:
  #   1. the longest dimension, if it exceeds the provided max_dimension.
  #   2. the shortest dimension, if it would be less than min_dimension after resizing based on longest dimension
  # Otherwise, returns the original image.
  def resize(self, img):
    width, height = img.width, img.height
    max_dimension, min_dimension = self.config.max_dimension, self.config.min_dimension
    if max(width, height) <= max_dimension or min(width, height) <= min_dimension:
      return img
    if width >= height:
      width = max_dimension
      height = int(height * (width / img.width))
      if height < min_dimension:
        height = min_dimension
        width = int(img.width * (min_dimension / img.height))
    else:
      height = max_dimension
      width = int(width * (height / img.height))
      if width < min_dimension:
        width = min_dimension
        height = int(img.height * (min_dimension / img.width))
    return img.resize((width, height))