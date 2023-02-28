import pytest
from PIL import Image
from src.utils.image_normalizer import ImageNormalizer
from src.utils.config import Config
from pathlib import Path

class TestImageNormalizer:
  def test_resize_to_max_dimension_no_change(self):
    test_img = Image.new('RGB', (256, 256))
    result = ImageNormalizer.resize_to_max_dimension(test_img, 512)
    assert result == test_img

  def test_resize_to_max_dimension_equal_high(self):
    test_img = Image.new('RGB', (1024, 1024))
    result = ImageNormalizer.resize_to_max_dimension(test_img, 512)
    assert result.width == 512
    assert result.height == 512

  def test_resize_to_max_dimension_both_high(self):
    test_img = Image.new('RGB', (1024, 768))
    result = ImageNormalizer.resize_to_max_dimension(test_img, 512)
    assert result.width == 512
    assert result.height == 384

  def test_resize_to_max_dimension_width_high(self):
    test_img = Image.new('RGB', (768, 500))
    result = ImageNormalizer.resize_to_max_dimension(test_img, 512)
    assert result.width == 512
    assert result.height == 333

  def test_resize_to_max_dimension_height_high(self):
    test_img = Image.new('RGB', (500, 900))
    result = ImageNormalizer.resize_to_max_dimension(test_img, 512)
    assert result.width == 284
    assert result.height == 512

  def test_build_output_path_jpg(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/norm/')
    subject = ImageNormalizer(config)
    src_path = Path('/path/to/src/base/sub/path/image.jpg')
    result = subject.build_output_path(src_path)
    assert result == Path('/path/to/output/base/norm/sub/path/image.jpg')

  def test_build_output_path_tiff(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/norm/')
    subject = ImageNormalizer(config)
    src_path = Path('/path/to/src/base/sub/path/image.TIFF')
    result = subject.build_output_path(src_path)
    assert result == Path('/path/to/output/base/norm/sub/path/image.jpg')

  def test_build_output_path_not_in_src(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/')
    subject = ImageNormalizer(config)
    src_path = Path('/path/to/another/path/image.jpg')
    with pytest.raises(ValueError):
      subject.build_output_path(src_path)

  def test_build_output_path_no_base(self):
    config = Config()
    config.output_base_path = Path('/path/to/output/base/norm/')
    subject = ImageNormalizer(config)
    src_path = Path('/path/to/src/base/sub/path/image.jpg')
    result = subject.build_output_path(src_path)
    assert result == Path('/path/to/output/base/norm/image.jpg')

  def test_process_unchanged(self, tmp_path):
    config = self.init_config(tmp_path)
    src_path = config.src_base_path / 'images/blue.jpg'
    src_path.parent.mkdir()
    src_img = Image.new('RGB', (500, 500), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)
    
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg')
    assert result.width == 500
    assert result.height == 500
    assert result.mode == 'RGB'

  def test_process_grayscale_tif(self, tmp_path):
    config = self.init_config(tmp_path)
    src_path = config.src_base_path / 'gray.tif'
    # L is grayscale color profile
    src_img = Image.new('L', (500, 500), 100)
    src_img.save(src_path, 'TIFF')
    subject = ImageNormalizer(config)
    
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'gray.jpg')
    assert result.width == 500
    assert result.height == 500
    assert result.mode == 'RGB'
    assert result.getpixel((0, 0)) == (100, 100, 100)

  def test_process_too_large(self, tmp_path):
    config = self.init_config(tmp_path)
    src_path = config.src_base_path / 'bigblue.jpg'
    src_img = Image.new('RGB', (5000, 4000), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)
    
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'bigblue.jpg')
    assert result.width == 512
    assert result.height == 409
    assert result.mode == 'RGB'

  def test_process_multi_runs(self, tmp_path):
    config = self.init_config(tmp_path)
    src_path = config.src_base_path / 'images/blue.jpg'
    src_path.parent.mkdir()
    src_img = Image.new('RGB', (500, 500), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)
    
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg')
    assert result.width == 500
    # Run process on image again with different dimensions, it should not change sizes since its already been generated
    config.max_dimension = 256
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg')
    assert result.width == 500
    # Run again with force flag, it should change this time
    config.force = True
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg')
    assert result.width == 256

  def init_config(self, tmp_path):
    config = Config()
    config.src_base_path = tmp_path / 'src/base'
    config.src_base_path.mkdir(parents=True)
    config.output_base_path = tmp_path / 'output'
    config.output_base_path.mkdir()
    config.max_dimension = 512
    return config