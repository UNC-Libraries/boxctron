from pathlib import Path
from src.utils.cropping_workflow_service import CroppingWorkflowService
from src.utils.classifier_config import ClassifierConfig
import pytest
from PIL import Image
from unittest.mock import patch

@pytest.fixture
def config(tmp_path):
  conf = ClassifierConfig()
  return conf

class TestCroppingWorkflowService:
  def test_process_defaults(self, tmp_path, config):
    csv_path = Path("fixtures/seg_report.csv")
    output_path = tmp_path / "cropped"
    config.src_base_path = Path('fixtures/normalized_images/').resolve()
    service = CroppingWorkflowService(csv_path, output_path, config)

    cropped_paths = service.process()

    assert 7 == len(cropped_paths)
    assert all(path.exists() for path in cropped_paths)
    # Cropped path contains the full path from the root of the file system, made relative to the output directory
    expected_path1 = output_path / "gilmer/00276_op0204_0001.jpg.jpg"
    assert expected_path1 in cropped_paths
    expected_path2 = output_path / "gilmer/00276_op0226a_0001.jpg.jpg"
    assert expected_path2 in cropped_paths

  def test_process_with_exclusions(self, tmp_path, config):
    csv_path = Path("fixtures/seg_report.csv")
    output_path = tmp_path / "cropped"
    config.src_base_path = Path('fixtures/normalized_images/').resolve()
    exclusions_path = tmp_path / "exclude.csv"
    with open(exclusions_path, 'w') as file:
      file.write('path,predicted_class,corrected_class\n')
      # Excluding path which would normally be cropped
      file.write('/gilmer/00276_op0204_0001.jpg,1,0\n')
      # Excluding path that would normally not be cropped
      file.write('/ncc/Cm912_1945u1_sheet1.jpg,1,0\n')
    service = CroppingWorkflowService(csv_path, output_path, config, exclusions_path = exclusions_path)

    cropped_paths = service.process()

    assert 6 == len(cropped_paths)
    assert all(path.exists() for path in cropped_paths)
    # Path was excluded path, so it should not have been cropped
    expected_path1 = output_path / "gilmer/00276_op0204_0001.jpg.jpg"
    assert expected_path1 not in cropped_paths
    assert not expected_path1.exists()
    expected_path2 = output_path / "gilmer/00276_op0226a_0001.jpg.jpg"
    assert expected_path2 in cropped_paths

  def test_crop_image_retries_after_metadata_save_error(self, tmp_path, config):
    source_path = tmp_path / 'source.tif'
    Image.new('RGB', (100, 100), 'red').save(source_path, 'TIFF')
    service = CroppingWorkflowService(tmp_path / 'report.csv', tmp_path / 'cropped', config)

    original_save = Image.Image.save
    call_count = 0

    def mock_save(self, *args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise TypeError("can't concat tuple to bytes")
      return original_save(self, *args, **kwargs)

    with patch.object(Image.Image, 'save', mock_save):
      result_path = service.crop_image(source_path, source_path, [0, 0, 0.5, 0.5])

    assert result_path.exists()
    with Image.open(result_path) as result:
      assert result.size == (50, 50)
      assert result.mode == 'RGB'
    assert call_count == 2

  def test_crop_image_retries_after_metadata_load_error(self, tmp_path, config):
    source_path = tmp_path / 'source.tif'
    Image.new('RGB', (100, 100), 'red').save(source_path, 'TIFF')
    service = CroppingWorkflowService(tmp_path / 'report.csv', tmp_path / 'cropped', config)

    original_crop = Image.Image.crop
    call_count = 0

    def mock_crop(self, *args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        self.info['xmp'] = ('malformed',)
        raise TypeError("expected string or bytes-like object, got 'tuple'")
      assert 'xmp' not in self.info
      return original_crop(self, *args, **kwargs)

    with patch.object(Image.Image, 'crop', mock_crop):
      result_path = service.crop_image(source_path, source_path, [0, 0, 0.5, 0.5])

    assert result_path.exists()
    with Image.open(result_path) as result:
      assert result.size == (50, 50)
      assert result.mode == 'RGB'
    assert call_count == 2
