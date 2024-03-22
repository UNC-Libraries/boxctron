from pathlib import Path
from src.utils.cropping_workflow_service import CroppingWorkflowService

class TestCroppingWorkflowService:
  def test_process_defaults(self, tmp_path):
    csv_path = Path("fixtures/seg_report.csv")
    output_path = tmp_path / "cropped"
    service = CroppingWorkflowService(csv_path, output_path)

    cropped_paths = service.process()

    assert 7 == len(cropped_paths)
    assert all(path.exists() for path in cropped_paths)
    # Cropped path contains the full path from the root of the file system, made relative to the output directory
    expected_path1 = output_path / str(Path("fixtures/normalized_images/gilmer/00276_op0204_0001.jpg").resolve())[1:]
    assert expected_path1 in cropped_paths
    expected_path2 = output_path / str(Path("fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg").resolve())[1:]
    assert expected_path2 in cropped_paths

  def test_process_original_relative_path(self, tmp_path):
    csv_path = Path("fixtures/seg_report.csv")
    output_path = tmp_path / "cropped"
    service = CroppingWorkflowService(csv_path, output_path, originals_base_path = Path('fixtures/normalized_images'))

    cropped_paths = service.process()

    assert 7 == len(cropped_paths)
    assert all(path.exists() for path in cropped_paths)
    # Cropped path is shortened since it was made relative to fixtures/normalized_images
    expected_path1 = output_path / "gilmer/00276_op0204_0001.jpg"
    assert expected_path1 in cropped_paths
    expected_path2 = output_path / "gilmer/00276_op0226a_0001.jpg"
    assert expected_path2 in cropped_paths

  def test_process_with_exclusions(self, tmp_path):
    csv_path = Path("fixtures/seg_report.csv")
    output_path = tmp_path / "cropped"
    exclusions_path = tmp_path / "exclude.csv"
    with open(exclusions_path, 'w') as file:
      file.write('path,predicted_class,corrected_class\n')
      # Excluding path which would normally be cropped
      file.write(f'{Path("fixtures/normalized_images/gilmer/00276_op0204_0001.jpg").resolve()},1,0\n')
      # Excluding path that would normally not be cropped
      file.write(f'{Path("fixtures/normalized_images/ncc/Cm912_1945u1_sheet1.jpg").resolve()},1,0\n')
    service = CroppingWorkflowService(csv_path, output_path, originals_base_path = Path('fixtures/normalized_images'), exclusions_path = exclusions_path)

    cropped_paths = service.process()

    assert 6 == len(cropped_paths)
    assert all(path.exists() for path in cropped_paths)
    # Path was excluded path, so it should not have been cropped
    expected_path1 = output_path / "gilmer/00276_op0204_0001.jpg"
    assert expected_path1 not in cropped_paths
    assert not expected_path1.exists()
    expected_path2 = output_path / "gilmer/00276_op0226a_0001.jpg"
    assert expected_path2 in cropped_paths
