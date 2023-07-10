import pytest
from PIL import Image
from src.utils.classifier_workflow_service import ClassifierWorkflowService
from src.utils.classifier_config import ClassifierConfig
from pathlib import Path
import torch
import os
import csv

@pytest.fixture
def config(tmp_path):
  conf = ClassifierConfig()
  conf.src_base_path = Path.cwd() / 'fixtures/normalized_images/'
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir(parents=True)
  conf.max_dimension = 256
  conf.min_dimension = 224
  conf.predict_rounding_threshold = 0.7
  conf.model_width = 256
  conf.resnet_depth = 18
  # Model was trained using 256 max dimensions, resnet18, over 20 epochs
  conf.model_path = Path.cwd() / 'fixtures/checkpoints/resnet18_256px.ckpt'
  return conf

class TestClassifierWorkflowService:
  def test_with_new_file(self, config, tmp_path):
    torch.manual_seed(42) 
    report_path = tmp_path / 'report.csv'
    img_paths = [Path.cwd() / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg',
        Path.cwd() / 'fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg',
        Path.cwd() / 'fixtures/normalized_images/ncc/Cm912m_U58b9.jpg',
        Path.cwd() / 'fixtures/normalized_images/ncc/fcpl_005.jpg']

    subject = ClassifierWorkflowService(config, report_path)
    subject.process(img_paths)

    assert Path.exists(report_path)
    
    with open(report_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    assert rows[1] == [str(img_paths[0]), str(config.output_base_path / 'gilmer/00276_op0204_0001.jpg'), '1', '0.8863']
    assert rows[2] == [str(img_paths[1]), str(config.output_base_path / 'ncc/G3902-F3-1981_U5_front.jpg'), '1', '0.9409']
    assert rows[3] == [str(img_paths[2]), str(config.output_base_path / 'ncc/Cm912m_U58b9.jpg'), '1', '0.9775']
    assert rows[4] == [str(img_paths[3]), str(config.output_base_path / 'ncc/fcpl_005.jpg'), '0', '0.6976']
