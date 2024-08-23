import os
import pytest
import csv
from pathlib import Path
from src.utils.segmentation_report_service import SegmentationReportService
from src.utils.json_utils import from_json

class TestSegmentationReportService:
  def test_generate(self, tmp_path):
    output_path = tmp_path / 'report'
    images_path = output_path / 'images'
    csv_path = Path('fixtures/seg_report.csv')
    service = SegmentationReportService(csv_path, output_path, norms_relative_path = os.getcwd())
    service.generate()    

    assert (output_path / 'report.html').is_file()
    data_wrapper = from_json(output_path / 'data.json')
    data = data_wrapper['data']
    assert data[0]['original'] == 'fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg'
    assert data[0]['pred_class'] == '0'
    assert data[0]['pred_conf'] == '0.0000'
    assert data[0]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg.jpg'
    assert data[0]['problem'] == False
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg.jpg').is_file()

    assert data[2]['original'] == 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg'
    assert data[2]['pred_class'] == '1'
    assert data[2]['pred_conf'] == '0.9990'
    assert data[2]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0204_0001.jpg.jpg'
    assert data[2]['problem'] == False
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg.jpg').is_file()

    assert data[3]['original'] == 'fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg'
    assert data[3]['pred_class'] == '1'
    assert data[3]['pred_conf'] == '0.9996'
    assert data[3]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg.jpg'
    assert data[3]['problem'] == True
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg.jpg').is_file()

    assert 14 == len(data)

    assert (output_path / 'data.csv').is_file()
    with open(output_path / 'data.csv', newline='') as f:
      reader = csv.reader(f)
      data = list(reader)

      assert len(data) == 15
      assert data[1][0].endswith('fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg')
      assert data[3][0].endswith('fixtures/normalized_images/gilmer/00276_op0204_0001.jpg')

  def test_generate_with_src_base_path(self, tmp_path):
    output_path = tmp_path / 'report'
    images_path = output_path / 'images'
    csv_path = Path('fixtures/seg_report.csv')
    src_base = Path('fixtures/normalized_images/')
    service = SegmentationReportService(csv_path, output_path, norms_relative_path = os.getcwd(), src_base_path = src_base)
    service.generate()

    assert (output_path / 'report.html').is_file()
    data_wrapper = from_json(output_path / 'data.json')
    data = data_wrapper['data']
    assert data[0]['original'] == '/gilmer/00276_op0217_0001_e.jpg'
    assert data[0]['pred_class'] == '0'
    assert data[0]['pred_conf'] == '0.0000'
    assert data[0]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg.jpg'
    assert data[0]['problem'] == False
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0217_0001_e.jpg.jpg').is_file()

    assert data[2]['original'] == '/gilmer/00276_op0204_0001.jpg'
    assert data[2]['pred_class'] == '1'
    assert data[2]['pred_conf'] == '0.9990'
    assert data[2]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0204_0001.jpg.jpg'
    assert data[2]['problem'] == False
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0204_0001.jpg.jpg').is_file()

    assert data[3]['original'] == '/gilmer/00276_op0226a_0001.jpg'
    assert data[3]['pred_class'] == '1'
    assert data[3]['pred_conf'] == '0.9996'
    assert data[3]['image'] == 'images/fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg.jpg'
    assert data[3]['problem'] == True
    assert (images_path / 'fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg.jpg').is_file()

    assert 14 == len(data)

    assert (output_path / 'data.csv').is_file()
    with open(output_path / 'data.csv', newline='') as f:
      reader = csv.reader(f)
      data = list(reader)

      assert len(data) == 15
      assert data[1][0] == '/gilmer/00276_op0217_0001_e.jpg'
      assert data[3][0] == '/gilmer/00276_op0204_0001.jpg'
