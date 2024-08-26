from pathlib import Path
from src.utils.cropping_workflow_service import CroppingWorkflowService
from src.utils.classifier_config import ClassifierConfig
import pytest

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
