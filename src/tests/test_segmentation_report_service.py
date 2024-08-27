import os
import pytest
from pathlib import Path
from src.utils.segmentation_report_service import SegmentationReportService
from src.utils.classifier_config import ClassifierConfig
from src.utils.json_utils import from_json
import shutil

@pytest.fixture
def config(tmp_path):
  conf = ClassifierConfig()
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir()
  conf.src_base_path = Path('fixtures/normalized_images/').resolve()
  return conf

class TestSegmentationReportService:
  def test_generate(self, tmp_path, config):
    norm_gilmer_path = config.output_base_path / 'gilmer'
    norm_gilmer_path.mkdir()

    self.copy_normalized_files(config.output_base_path)

    report_path = tmp_path / 'report'
    images_path = report_path / 'images'
    csv_path = Path('fixtures/seg_report.csv')
    service = SegmentationReportService(csv_path, report_path, config)
    service.generate()

    assert (report_path / 'report.html').is_file()
    assert (report_path / 'data.csv').is_file()
    data_wrapper = from_json(report_path / 'data.json')
    data = data_wrapper['data']
    assert data[0]['original'] == '/gilmer/00276_op0217_0001_e.jpg'
    assert data[0]['pred_class'] == '0'
    assert data[0]['pred_conf'] == '0.0000'
    assert data[0]['image'] == 'images/gilmer/00276_op0217_0001_e.jpg.jpg'
    assert data[0]['problem'] == False
    assert (images_path / 'gilmer/00276_op0217_0001_e.jpg.jpg').is_file()

    assert data[2]['original'] == '/gilmer/00276_op0204_0001.jpg'
    assert data[2]['pred_class'] == '1'
    assert data[2]['pred_conf'] == '0.9990'
    assert data[2]['image'] == 'images/gilmer/00276_op0204_0001.jpg.jpg'
    assert data[2]['problem'] == False
    assert (images_path / 'gilmer/00276_op0204_0001.jpg.jpg').is_file()

    assert data[3]['original'] == '/gilmer/00276_op0226a_0001.jpg'
    assert data[3]['pred_class'] == '1'
    assert data[3]['pred_conf'] == '0.9996'
    assert data[3]['image'] == 'images/gilmer/00276_op0226a_0001.jpg.jpg'
    assert data[3]['problem'] == True
    assert (images_path / 'gilmer/00276_op0226a_0001.jpg.jpg').is_file()

    assert 14 == len(data)

  def copy_normalized_files(self, output_base_path):
    source_dir = Path("fixtures/normalized_images")

    # Iterate over all files in the source directory and its subdirectories
    for file_path in source_dir.rglob('*'):
      if file_path.is_file():  # Ensure it's a file, not a directory
        # Create the target path by appending .jpg to the file name
        target_path = output_base_path / file_path.relative_to(source_dir)
        target_path = target_path.with_name(target_path.name + ".jpg")

        # Ensure the target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file to the new path with the modified name
        shutil.copy2(file_path, target_path)
