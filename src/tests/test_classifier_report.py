import os
import pytest
import shutil
from src.utils.classifier_report import DataParser, ReportGenerator

class TestDataParser:
        
    def test_simple_csv(self):
        parser = DataParser("fixtures/sample_report.csv")
        data = parser.get_data()
        # CSV has 5 items
        assert len(data) == 50
        # items should be in a list
        assert type(data) == list
        # check two items
        item_1 = data[0]
        item_2 = data[1]
        # check item 1
        assert type(item_1) == dict
        assert item_1['original_path'] == "root_dir/subdir_3/gilmer/00276_op0226a_0001.jpg"
        assert item_1['normalized_path'] == "fixtures/normalized_images/gilmer/00276_op0226a_0001.jpg"
        assert item_1['predicted_class'] == "0"
        assert item_1['predicted_conf'] == "0.7218731549949939"
        #check item 2
        assert type(item_2) == dict
        assert item_2['original_path'] == "root_dir/subdir_1/sfc/00276_op0204_0001.jpg"
        assert item_2['normalized_path'] == "fixtures/normalized_images/gilmer/00276_op0204_0001.jpg"
        assert item_2['predicted_class'] == "0"
        assert item_2['predicted_conf'] == "0.6113711347206616"
    
    def test_normalize_url(self):
        http_url = "https://example.com/shared"
        substring = '/normalized_images/'
        parser = DataParser("fixtures/sample_report.csv", http_url, substring)
        data = parser.get_data()
        # check both items' normalized path
        item_1 = data[0]
        item_2 = data[1]
        assert item_1['normalized_path'] == "https://example.com/shared/normalized_images/gilmer/00276_op0226a_0001.jpg"
        assert item_2['normalized_path'] == "https://example.com/shared/normalized_images/gilmer/00276_op0204_0001.jpg"
    
    def test_create_stats(self):
        parser = DataParser("fixtures/sample_report.csv")
        stats = parser.get_stats()
        
        assert type(stats) == list
        assert len(stats) == 16
        
        item_1 = stats[1]
        item_2 = stats[-2]
        
        assert item_1['path'] == 'root_dir/subdir_3'
        assert item_1['count'] == 13
        assert item_1['has_CB'] == 'True'
        assert item_1['count_CB'] == 6
        assert item_1['percent_CB'] == 0.46153846153846156
        assert item_1['avg_conf_CB'] == 0.4779377137821212
        
        assert item_2['path'] == 'root_dir/subdir_3/sfc'
        assert item_2['count'] == 2
        assert item_2['has_CB'] == 'False'
        assert item_2['count_CB'] == 0
        assert item_2['percent_CB'] == 0
        assert item_2['avg_conf_CB'] == 0

class TestReportGenerator:
    
    # check the report is saved in the correct path
    def test_report_path(self, tmp_path):
        parser = DataParser("fixtures/sample_report.csv")
        data = parser.get_data()
        generator = ReportGenerator()
        generator.create_html_page(data)
        # create temp directory
        temp = tmp_path / "temp"
        temp.mkdir()
        report_path = temp / "report.html"
        generator.save_file(report_path)
        # check if file exists
        assert os.path.exists(report_path)
        
