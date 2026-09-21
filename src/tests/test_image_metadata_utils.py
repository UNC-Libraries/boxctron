import pytest
from PIL import Image
from src.utils.image_metadata_utils import retry_after_stripping_metadata


def test_retry_after_stripping_metadata_retries_operation():
  image = Image.new('RGB', (10, 10))
  image.info['xmp'] = ('malformed',)
  image.info['icc_profile'] = b'profile'
  image.info['exif'] = b'exif'
  call_count = 0

  def operation():
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise TypeError("expected string or bytes-like object, got 'tuple'")
    return 'success'

  assert retry_after_stripping_metadata(image, operation) == 'success'
  assert call_count == 2
  assert 'xmp' not in image.info
  assert 'icc_profile' not in image.info
  assert 'exif' not in image.info


def test_retry_after_stripping_metadata_does_not_handle_other_errors():
  image = Image.new('RGB', (10, 10))

  def operation():
    raise ValueError('unrelated error')

  with pytest.raises(ValueError, match='unrelated error'):
    retry_after_stripping_metadata(image, operation)
