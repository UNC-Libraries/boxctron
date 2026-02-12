import pytest
from PIL import Image
from src.utils.image_normalizer import ImageNormalizer
from src.utils.config import Config
from pathlib import Path
from unittest.mock import patch
import shutil

@pytest.fixture
def config(tmp_path):
  conf = Config()
  conf.src_base_path = tmp_path / 'src/base'
  conf.src_base_path.mkdir(parents=True)
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir()
  conf.max_dimension = 512
  conf.min_dimension = 224
  return conf

class TestImageNormalizer:
  def test_resize_longest_less_than_max(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (256, 256))
    result = subject.resize(test_img)
    assert result == test_img

  def test_resize_shortest_less_than_min(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (768, 224))
    result = subject.resize(test_img)
    assert result == test_img

  def test_resize_both_equally_too_long(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (1024, 1024))
    result = subject.resize(test_img)
    assert result.width == 512
    assert result.height == 512

  def test_resize_both_too_long(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (1024, 768))
    result = subject.resize(test_img)
    assert result.width == 512
    assert result.height == 384

  def test_resize_width_high(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (768, 500))
    result = subject.resize(test_img)
    assert result.width == 512
    assert result.height == 333

  def test_resize_height_high(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (500, 900))
    result = subject.resize(test_img)
    assert result.width == 284
    assert result.height == 512

  def test_resize_height_less_than_min_after_resize(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (768, 256))
    result = subject.resize(test_img)
    assert result.width == 672
    assert result.height == 224

  def test_resize_width_less_than_min_after_resize(self, config):
    subject = ImageNormalizer(config)
    test_img = Image.new('RGB', (256, 768))
    result = subject.resize(test_img)
    assert result.width == 224
    assert result.height == 672

  def test_build_output_path_jpg(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/norm/')
    subject = ImageNormalizer(config)
    src_path = Path('/sub/path/image.jpg')
    result = subject.build_output_path(src_path)
    assert result == Path('/path/to/output/base/norm/sub/path/image.jpg.jpg')

  def test_build_output_path_tiff(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/norm/')
    subject = ImageNormalizer(config)
    src_path = Path('/sub/path/image.TIFF')
    result = subject.build_output_path(src_path)
    assert result == Path('/path/to/output/base/norm/sub/path/image.TIFF.jpg')

  def test_process_not_in_src(self):
    config = Config()
    config.src_base_path = Path('/path/to/src/base/')
    config.output_base_path = Path('/path/to/output/base/')
    subject = ImageNormalizer(config)
    src_path = Path('/path/to/another/path/image.jpg')
    with pytest.raises(ValueError):
      subject.process(src_path)

  def test_process_no_src_base(self, config):
    src_base_path = config.src_base_path
    config.src_base_path = None
    subject = ImageNormalizer(config)
    src_path = src_base_path / 'images/blue.jpg'
    src_path.parent.mkdir()
    src_img = Image.new('RGB', (500, 500), 'blue')
    src_img.save(src_path)
    result = subject.process(src_path)
    assert result == config.output_base_path / (str(src_base_path)[1:] + '/images/blue.jpg.jpg')

  def test_process_unchanged(self, config):
    src_path = config.src_base_path / 'images/blue.jpg'
    src_path.parent.mkdir()
    src_img = Image.new('RGB', (500, 500), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)

    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg.jpg')
    assert result.width == 500
    assert result.height == 500
    assert result.mode == 'RGB'

  def test_process_grayscale_tif(self, config):
    src_path = config.src_base_path / 'gray.tif'
    # L is grayscale color profile
    src_img = Image.new('L', (500, 500), 100)
    src_img.save(src_path, 'TIFF')
    subject = ImageNormalizer(config)

    subject.process(src_path)
    result = Image.open(config.output_base_path / 'gray.tif.jpg')
    assert result.width == 500
    assert result.height == 500
    assert result.mode == 'RGB'
    assert result.getpixel((0, 0)) == (100, 100, 100)

  def test_process_too_large(self, config):
    src_path = config.src_base_path / 'bigblue.jpg'
    src_img = Image.new('RGB', (5000, 4000), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)

    subject.process(src_path)
    result = Image.open(config.output_base_path / 'bigblue.jpg.jpg')
    assert result.width == 512
    assert result.height == 409
    assert result.mode == 'RGB'

  def test_process_multi_runs(self, config):
    src_path = config.src_base_path / 'images/blue.jpg'
    src_path.parent.mkdir()
    src_img = Image.new('RGB', (500, 500), 'blue')
    src_img.save(src_path)
    subject = ImageNormalizer(config)

    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg.jpg')
    assert result.width == 500
    # Run process on image again with different dimensions, it should not change sizes since its already been generated
    config.max_dimension = 256
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg.jpg')
    assert result.width == 500
    # Run again with force flag, it should change this time
    config.force = True
    subject.process(src_path)
    result = Image.open(config.output_base_path / 'images/blue.jpg.jpg')
    assert result.width == 256

  def test_process_malformed_jp2(self, config):
    src_path = config.src_base_path / 'malformed.jp2'
    shutil.copyfile(Path('fixtures/source_images/malformed.jp2'), src_path)
    subject = ImageNormalizer(config)

    subject.process(src_path)
    result = Image.open(config.output_base_path / 'malformed.jp2.jpg')
    assert result.width == 565
    assert result.height == 224
    assert result.mode == 'RGB'

  # Test that files with problematic XMP metadata that cause TypeError
  # are handled by stripping metadata and retrying
  def test_process_with_xmp_metadata_error(self, config):
    src_path = config.src_base_path / 'tiff_with_xmp.tif'
    src_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a test image with XMP metadata
    src_img = Image.new('RGB', (500, 500), 'red')
    src_img.info['xmp'] = b'<test>xmp data</test>'
    src_img.info['icc_profile'] = b'fake_icc_profile'
    src_img.save(src_path, 'TIFF')

    subject = ImageNormalizer(config)

    # Mock the save method to raise TypeError on first call (simulating XMP bug),
    # then call the real save method on subsequent calls
    original_save = Image.Image.save
    call_count = 0

    def mock_save(self, *args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        # Simulate the PIL XMP concatenation bug
        raise TypeError("can't concat tuple to bytes")
      # On retry, call the real save method
      return original_save(self, *args, **kwargs)

    with patch.object(Image.Image, 'save', mock_save):
      result_path = subject.process(src_path)

    # Verify the file was created successfully after retry
    assert result_path.exists()
    result = Image.open(result_path)
    assert result.width == 500
    assert result.height == 500
    assert result.mode == 'RGB'
    assert call_count == 2  # First call failed, second succeeded
