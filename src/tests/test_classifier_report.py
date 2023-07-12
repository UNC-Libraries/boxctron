import os
import shutil
from src.utils.classifier_report import ReportGenerator

class TestReportGenerator:
        
    def test_simple_csv(self):
        
        generator = ReportGenerator()
        generator.parse_csv("fixtures/report_test.csv")
        
        # CSV has 5 items
        assert len(generator.data) == 5
        
        # items should be in a list
        assert type(generator.data) == list
        
        # check two items
        item_1 = generator.data[0]
        item_2 = generator.data[1]
        
        # check item 1
        assert type(item_1) == dict
        assert item_1['original_path'] == "http://localhost:8081/ncc/Cm912_1945u1_sheet1.jpg"
        assert item_1['normalized_path'] == "/ml-repo-preingest-processing/fixtures/normalized_images/ncc/Cm912_1945u1_sheet1.jpg"
        assert item_1['predicted_class'] == "1"
        assert item_1['predicted_conf'] == "0.523"
        
        #check item 2
        assert type(item_2) == dict
        assert item_2['original_path'] == "http://localhost:8081/ncc/P0004_0483_0001_verso.jpg"
        assert item_2['normalized_path'] == "/ml-repo-preingest-processing/fixtures/normalized_images/ncc/P0004_0483_0001_verso.jpg"
        assert item_2['predicted_class'] == "1"
        assert item_2['predicted_conf'] == "0.823"
    
        
    def test_normalize_url(self):
        
        http_url = "https://dcr-test.lib.cunc.edu/shared/"
        
        generator = ReportGenerator()
        generator.parse_csv("fixtures/report_test.csv")
        
        generator.normalize_urls(http_url)
        
        # check two rows
        item_1 = generator.data[0]
        item_2 = generator.data[1]
        
        assert item_1['normalized_path'] == "https://dcr-test.lib.cunc.edu/shared/Cm912_1945u1_sheet1.jpg"
        assert item_2['normalized_path'] == "https://dcr-test.lib.cunc.edu/shared/P0004_0483_0001_verso.jpg"
        
    # check the report is saved in the correct path
    def test_report_path(self):
        
        generator = ReportGenerator()
        generator.parse_csv("fixtures/report_test.csv")
        generator.create_html_page()
        
        # creates temp directory
        os.mkdir("./temp")
        save_path = "./temp/report.html"
        
        generator.save_file(save_path)
        
        assert os.path.exists(save_path)
        
        # deletes temp directory and content
        shutil.rmtree("./temp")
        
        
    
    
    
    