import pytest
from PIL import Image
from src.utils.classifier_workflow_service import ClassifierWorkflowService
from src.utils.classifier_config import ClassifierConfig
from pathlib import Path
import torch
import os
import csv
import math

@pytest.fixture
def config(tmp_path):
  conf = ClassifierConfig()
  conf.src_base_path = Path.cwd() / 'fixtures/normalized_images/'
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir(parents=True)
  conf.progress_log_path = tmp_path / 'progress.log'
  conf.max_dimension = 256
  conf.min_dimension = 224
  conf.predict_rounding_threshold = 0.75
  conf.model_width = 256
  conf.resnet_depth = 18
  # Model was trained using 256 max dimensions, resnet18, over 20 epochs
  conf.model_path = Path.cwd() / 'fixtures/checkpoints/resnet18_256px.ckpt'
  return conf

class TestClassifierWorkflowService:
  # def test_with_new_file(self, config, tmp_path):
  #   torch.manual_seed(42) 
  #   report_path = tmp_path / 'report.csv'
  #   img_paths = [Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/Cm912m_U58b9.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/fcpl_005.jpg']

  #   subject = ClassifierWorkflowService(config, report_path)
  #   subject.process(img_paths)

  #   assert Path.exists(report_path)

  #   rows = self.load_csv_rows(report_path)
  #   self.assert_row_matches(rows[1], img_paths[0], config.output_base_path / 'gilmer/00276_op0204_0001.jpg', '1', 0.8863)
  #   self.assert_row_matches(rows[2], img_paths[1], config.output_base_path / 'ncc/G3902-F3-1981_U5_front.jpg', '1', 0.9409)
  #   self.assert_row_matches(rows[3], img_paths[2], config.output_base_path / 'ncc/Cm912m_U58b9.jpg', '1', 0.9775)
  #   self.assert_row_matches(rows[4], img_paths[3], config.output_base_path / 'ncc/fcpl_005.jpg', '0', 0.6976)
  #   assert len(rows) == 5 # includes header row

  # def test_with_resume(self, config, tmp_path):
  #   torch.manual_seed(42) 
  #   report_path = tmp_path / 'report.csv'
  #   img_paths = [Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg']

  #   subject = ClassifierWorkflowService(config, report_path)
  #   subject.process(img_paths)

  #   rows = self.load_csv_rows(report_path)
  #   self.assert_row_matches(rows[1], img_paths[0], config.output_base_path / 'gilmer/00276_op0204_0001.jpg', '1', 0.8863)
  #   self.assert_row_matches(rows[2], img_paths[1], config.output_base_path / 'ncc/G3902-F3-1981_U5_front.jpg', '1', 0.9409)
  #   assert len(rows) == 3 # includes header row

  #   progress_list = self.load_progress_list(tmp_path)
  #   assert progress_list[0] == str(img_paths[0])
  #   assert progress_list[1] == str(img_paths[1])
  #   assert len(progress_list) == 2

  #   img_paths2 = [Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/Cm912m_U58b9.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/ncc/fcpl_005.jpg']

  #   subject2 = ClassifierWorkflowService(config, report_path)
  #   subject2.process(img_paths2)

  #   rows = self.load_csv_rows(report_path)
  #   self.assert_row_matches(rows[1], img_paths2[0], config.output_base_path / 'gilmer/00276_op0204_0001.jpg', '1', 0.8863)
  #   self.assert_row_matches(rows[2], img_paths2[1], config.output_base_path / 'ncc/G3902-F3-1981_U5_front.jpg', '1', 0.9409)
  #   self.assert_row_matches(rows[3], img_paths2[2], config.output_base_path / 'ncc/Cm912m_U58b9.jpg', '1', 0.9775)
  #   self.assert_row_matches(rows[4], img_paths2[3], config.output_base_path / 'ncc/fcpl_005.jpg', '0', 0.6976)
  #   assert len(rows) == 5 # includes header row

  #   progress_list = self.load_progress_list(tmp_path)
  #   assert progress_list[0] == str(img_paths2[0])
  #   assert progress_list[1] == str(img_paths2[1])
  #   assert progress_list[2] == str(img_paths2[2])
  #   assert progress_list[3] == str(img_paths2[3])
  #   assert len(progress_list) == 4

  # def test_restart(self, config, tmp_path):
  #   torch.manual_seed(42) 
  #   report_path = tmp_path / 'report.csv'
  #   img_paths = [Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg']

  #   subject = ClassifierWorkflowService(config, report_path)
  #   subject.process(img_paths)

  #   img_paths2 = [Path.cwd() / 'fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg',
  #       Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg']

  #   subject2 = ClassifierWorkflowService(config, report_path, restart = True)
  #   subject2.process(img_paths2)

  #   rows = self.load_csv_rows(report_path)
  #   # After the restart, both entries should be present and the item that was previous first should be second to prove we didn't simply add to the CSV
  #   assert len(rows) == 3 # includes header row
  #   self.assert_row_matches(rows[1], img_paths2[0], config.output_base_path / 'ncc/G3902-F3-1981_U5_front.jpg', '1', 0.9409)
  #   self.assert_row_matches(rows[2], img_paths2[1], config.output_base_path / 'gilmer/00276_op0204_0001.jpg', '1', 0.8863)

  #   # Progress log should have 2 items
  #   progress_list = self.load_progress_list(tmp_path)
  #   assert progress_list[0] == str(img_paths2[0])
  #   assert progress_list[1] == str(img_paths2[1])
  #   assert len(progress_list) == 2

  def assert_row_matches(self, row, exp_original_path, exp_norm_path, exp_class, exp_conf):
    assert row[0] == str(exp_original_path)
    assert row[1] == str(exp_norm_path)
    assert row[2] == exp_class
    # allow confidence to be within 5% of expected value, since the number isn't totally consistent across environments
    assert math.isclose(float(row[3]), exp_conf, rel_tol=0.05)

  def load_csv_rows(self, report_path):
    with open(report_path, 'r') as file:
      reader = csv.reader(file)
      return list(reader)

  def load_progress_list(self, tmp_path):
    with open(tmp_path / 'progress.log', 'r') as f:
      return list(line for line in f.read().splitlines())
