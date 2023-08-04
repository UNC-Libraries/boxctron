from airium import Airium
from pathlib import Path
from csv import DictReader
import webbrowser
import argparse
import os
import re


class ReportGenerator:

    # parses csv and creates a list of dictionaries  
    def parse_csv(self, csv_path):
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            self.data = list(DictReader(f))
            
    def normalize_urls(self, http_url, substring):
        for item in self.data:
            item["normalized_path"] = re.sub(rf'.+(?={substring})', http_url, item["normalized_path"])

    # creates the HTML page
    def create_html_page(self):
        a = Airium()
        a('<!DOCTYPE html>')
        with a.html(lang='en'):
            with a.head():
                a.meta(charset='utf-8')
                a.title(_t='Classifier Results')

                # CSS CDNs
                a.link(rel="stylesheet", href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/searchpanes/2.1.2/css/searchPanes.dataTables.min.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/select/1.6.2/css/select.dataTables.min.css")
            
            # creates table element
            with a.body():
                a.table(id="myTable", klass="display")
                
                # JQuery Core and UI CDNs
                a.script(src="https://code.jquery.com/jquery-3.7.0.js", integrity="sha256-JlqSTELeR4TLqP0OG9dxM7yDPqX1ox/HfgiSLBj8+kM=", crossorigin="anonymous")
                a.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.js", integrity="sha256-xLD7nhI62fcsEZK2/v8LsBcb4lG7dgULkuXoXB/j91c=", crossorigin="anonymous")

                # Datables CDNs
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
                                {{ title: 'Image', data: 'normalized_path', render: (d,t,r,m) => '<img src="'+d+'" style=height:200px; loading="lazy" />'}},
                                {{ title: 'Path', data: 'original_path'}},
                                {{ title: 'Class', data: 'predicted_class'}},
                                {{ title: 'Confidence', data: 'predicted_conf'}}
                            ],
                        }});
                    ''')
        
        self.html_page = a
    
    # save file in predefined reports folder or into the defined directory
    def save_file(self, output_path):
       
        html = str(self.html_page)
        
        # if save path was not specified, creates reports directory and reports.html file within it
        if output_path:
             with open(output_path, 'w') as f:
                f.write(html)
        else:
            dirname = os.getcwd()
            report_dir = os.path.join(dirname, 'reports/')
            
            if not os.path.exists(report_dir):
                os.mkdir(report_dir)
                print('directory has been created')
            
            output_path = os.path.join(report_dir, "report.html")
            
            with open(output_path, 'w') as f:
                f.write(html)            
        
        print(f"HTML file saved at {output_path}")
        self.url = f"file://{os.path.abspath(output_path)}"



    # launches the HTML page in default browser
    def launch_page(self):
        webbrowser.open_new_tab(self.url)
