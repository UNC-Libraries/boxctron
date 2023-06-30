# this command-line tool can take two arguments which must be preceded by their respective flags
#   1) -f : (REQUIRED) The CSV file that has four columns: 
#                      original_path, normalized_path, predicted_class, predicted_conf
#   2) -s : (OPTIONAL) The path to the saved html path. If this argument is not specified, 
#                      a 'reports' directory will be created where 'reports.html' will be saved.


from airium import Airium
from pathlib import Path
from csv import DictReader
import webbrowser
import argparse
import os


class ReportGenerator:
    def __init__(self):
        self.parse_args()
        self.parse_csv()
        self.create_html_page()
        self.launch_page()
    
    # parses command-line arguments and checks for proper file extensions
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_path', '-f', type=Path, required=True, help="Path to csv file that will be used to generate html page.")
        parser.add_argument('--save_path', '-s', type=Path, required=False, help="If path is provided, the html page generated will be saved to that path.")
        p = parser.parse_args()
        
        assert os.path.splitext(p.file_path)[-1].lower() == '.csv'
        
        if p.save_path:
            assert os.path.splitext(p.save_path)[-1].lower() == '.html'

        self.csv_path = p.file_path
        
        try:
            self.save_path = p.save_path
        except:
            self.save_path = False
 
    # parses csv and creates a list of dictionaries  
    def parse_csv(self):
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            self.data = list(DictReader(f))
    
    # creates the HTML page
    def create_html_page(self):
        a = Airium()
        a('<!DOCTYPE html>')
        with a.html(lang='en'):
            with a.head():
                a.meta(charset='utf-8')
                a.title(_t='Classifier Results')
                
                # CSS CDMs
                a.link(rel="stylesheet", href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/searchpanes/2.1.2/css/searchPanes.dataTables.min.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/select/1.6.2/css/select.dataTables.min.css")
            
            # creates table element
            with a.body():
                a.table(id="myTable", klass="display")
                
                # JQuery Core and UI CDN
                a.script(src="https://code.jquery.com/jquery-3.7.0.js", integrity="sha256-JlqSTELeR4TLqP0OG9dxM7yDPqX1ox/HfgiSLBj8+kM=", crossorigin="anonymous")
                a.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.js", integrity="sha256-xLD7nhI62fcsEZK2/v8LsBcb4lG7dgULkuXoXB/j91c=", crossorigin="anonymous")

                # Datables CDN
                a.script(src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js")
                a.script(src="https://cdn.datatables.net/searchpanes/2.1.2/js/dataTables.searchPanes.min.js")
                a.script(src="https://cdn.datatables.net/select/1.6.2/js/dataTables.select.min.js")
                a.script(src="https://cdn.datatables.net/fixedheader/3.3.2/js/dataTables.fixedHeader.min.js")
                a.script(src="https://cdn.datatables.net/responsive/2.4.1/js/dataTables.responsive.min.js")
                
                # Javascript creating the table
                #  the 'Path' column accounts for the csv 
                with a.script():
                    a(f'''
                        let table = new DataTable('#myTable', {{
                            data: {self.data},
                            responsive: true,
                            fixedHeader:{{ header: true, footer: true }},
                            ordering: true,
                            paging: true,
                            scrollY: true,
                            searching: true,
                            searchPanes: {{viewTotal: true, layout: 'columns-4', initCollapsed: true}},
                            dom: 'Plfrtip',
                            columns: [
                                {{ title: 'Image', data: 'normalized_path', render: (d,t,r,m) => '<img src="'+d+'" style=height:200px; />'}},
                                {{ title: 'Path', data: 'original_path'}},
                                {{ title: 'Class', data: 'predicted_class'}},
                                {{ title: 'Confidence', data: 'predicted_conf'}}
                            ],
                        }});
                    ''')
        
        self.html_page = a
    
    # launches the HTML page in default browser
    def launch_page(self):
        
        html = str(self.html_page)
        
        # if save path was not specified, creates reports directory and reports.html file within it
        if not self.save_path:
            dirname = os.path.dirname(os.getcwd())
            report_dir = os.path.join(dirname, 'reports/')
            
            if not os.path.exists(report_dir):
                os.mkdir(report_dir)
                
            self.save_path = os.path.join(report_dir, 'report.html')
        
        url = f"file://{os.path.abspath(self.save_path)}"

        with open(self.save_path, 'w') as f:
            f.write(html)

        webbrowser.open_new_tab(url)


ReportGenerator()
