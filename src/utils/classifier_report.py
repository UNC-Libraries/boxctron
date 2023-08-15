from airium import Airium
from collections import defaultdict
from csv import DictReader
import os
import re
import webbrowser

class DataParser:
    def __init__(self, csv_path, url=False, substring=False):
        self.csv = csv_path
        self.url = url
        self.data = False
        self.substring = substring
    
    # parses csv and creates a list of dictionaries  
    def parse_csv(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
             self.data = list(DictReader(f))
    
    # converts imge paths to urls based on argumet
    def normalize_to_url(self, http_url, substring):
        for item in self.data:
            item["normalized_path"] = re.sub(rf'.+(?={substring})', http_url, item["normalized_path"])

    # if mode==1 generator constructs statistics about dir tree in files
    def create_stats(self):
        self.stats = defaultdict(lambda: {'path': '', 'count': 0, 'count_CB': 0, 'percent_CB': 0, 'has_CB': 'False', 'avg_conf_CB': 0}) 
        for item in self.data:
            folders = item['original_path'].lstrip('/').split('/')
            current_path = ''
            for folder in folders[:-1]:
                if current_path:
                    current_path += f'/{folder}'
                else:
                    current_path = folder
                self.stats[current_path]['path'] = current_path
                self.stats[current_path]['count'] +=1
                if item['predicted_class'] == '1':
                    self.stats[current_path]['count_CB'] += 1
                    self.stats[current_path]['has_CB'] = 'True'
                    self.stats[current_path]['avg_conf_CB'] += float(item['predicted_conf'])
        
        for k, stats, in self.stats.items():
            if stats['has_CB'] == 'True':
                stats['avg_conf_CB'] = stats['avg_conf_CB'] / (stats['count_CB'])
                stats['percent_CB'] = stats['count_CB'] / stats['count']
    # returns item-level data
    def get_data(self):
        self.parse_csv(self.csv)
        if self.url:
            self.normalize_to_url(self.url, self.substring)
        return self.data
    # returns aggregate data
    def get_stats(self):
        if not self.data:
            self.parse_csv(self.csv)
        self.create_stats()
        return list(self.stats.values())


class ReportGenerator:
    # creates the HTML page
    def create_html_page(self, data, stats=False):
        a = Airium()
        a('<!DOCTYPE html>')
        with a.html(lang='en'):
            # header info
            with a.head():
                a.meta(charset='UTF-8')
                a.meta(name="viewport", content="width=device-width, initial-scale=1.0")
                a.title(_t="Model report")
                a.title(_t='Classifier Results')
                # CSS CDNs
                a.link(rel="stylesheet", href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/searchpanes/2.1.2/css/searchPanes.dataTables.min.css")
                a.link(rel="stylesheet", href="https://cdn.datatables.net/select/1.6.2/css/select.dataTables.min.css")
            # creates table elements
            with a.body():
                if stats:
                    a.button(id='toggleButton', _t="See Images Report")
                    with a.div(id='statsTable-container', klass='table-container'):
                        a.table(id='statsTable', klass='display')
                        with a.thead(): pass
                with a.div(id='imagesTable-container', klass='table-container', style='display:none;' if stats else ''):
                    a.table(id='imagesTable', klass='display')
                    with a.thead(): pass
                       
                # JQuery Core and UI CDNs
                a.script(src="https://code.jquery.com/jquery-3.7.0.js", integrity="sha256-JlqSTELeR4TLqP0OG9dxM7yDPqX1ox/HfgiSLBj8+kM=", crossorigin="anonymous")
                a.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.js", integrity="sha256-xLD7nhI62fcsEZK2/v8LsBcb4lG7dgULkuXoXB/j91c=", crossorigin="anonymous")
                # Datables CDNs
                a.script(src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js")
                a.script(src="https://cdn.datatables.net/searchpanes/2.1.2/js/dataTables.searchPanes.min.js")
                a.script(src="https://cdn.datatables.net/select/1.6.2/js/dataTables.select.min.js")
                a.script(src="https://cdn.datatables.net/fixedheader/3.3.2/js/dataTables.fixedHeader.min.js")
                a.script(src="https://cdn.datatables.net/responsive/2.4.1/js/dataTables.responsive.min.js")
                
                # Javascript creating the table the 'Path' column accounts for the csv 
                with a.script():
                    a("$(document).ready( () => {")
                    a(f'''
                        $("#imagesTable").DataTable({{
                            data: {data},
                            renderer: "jquery",
                            deferRender: true,
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
                                {{ title: 'Confidence', data: 'predicted_conf', render: $.fn.dataTable.render.number(',', '.', 3, '')}}
                            ],
                        }});
                    ''')
                    if stats:
                        a(f'''
                            $("#statsTable").DataTable({{
                                data: {stats},
                                renderer: "jquery",
                                deferRender: true,
                                responsive: true,
                                fixedHeader:{{ header: true, footer: true }},
                                ordering: true,
                                paging: true,
                                scrollY: true,
                                searching: true,
                                searchPanes: {{viewTotal: true, columns: [2], layout: 'columns-4', initCollapsed: true}},
                                dom: 'Plfrtip',
                                columns: [
                                    {{ title: 'Directory', data: 'path', class: 'hasPointer', 
                                        createdCell: (td, cData, rData, row, col) => {{
                                            $(td).css({{"cursor":"pointer", "text-decoration": "underline"}});
                                            $(td).click( ()=> {{ filterTable(cData); }});       
                                            }}
                                    }},
                                    {{ title: 'Total Images', data: 'count'}},
                                    {{ title: 'Contains CB', data: 'has_CB'}},
                                    {{ title: 'Total CB', data: 'count_CB'}},
                                    {{ title: 'Percent CB', data: 'percent_CB', render: $.fn.dataTable.render.number(',', '.', 3, '')}},
                                    {{ title: 'Average CB Conf', data: 'avg_conf_CB', render: $.fn.dataTable.render.number(',', '.', 3, '')}},
                                        ],
                                }});
                        ''')
                        a('''
                        $("#toggleButton").click( () => {
                            $("#imagesTable-container").toggle();
                            $("#statsTable-container").toggle();
                            $("#imagesTable").DataTable().search('').draw();
                            if ($("#toggleButton").text() === "See Images Report") {
                                $("#toggleButton").text("See Aggregate Report");
                            } else {
                                $("#toggleButton").text("See Images Report");
                            }
                        });
                    ''')
                        a('''
                        filterTable = (filter_str) => {{
                            $("#statsTable-container").toggle();
                            $("#imagesTable").DataTable().search(filter_str).draw();
                            $("#imagesTable-container").toggle();}}
                          ''')
                    a("});")
        self.html = str(a)
    # save file in predefined reports folder or into the defined directory
    def save_file(self, output_path):
        # if save path was not specified, creates reports directory and reports.html file within it
        if not output_path:
            dirname = os.getcwd()
            report_dir = os.path.join(dirname, 'reports/')
            if not os.path.exists(report_dir):
                os.mkdir(report_dir)
                print('directory has been created')
            output_path = os.path.join(report_dir, "report.html")
        # saves file
        with open(output_path, 'w') as f:
            f.write(self.html)            
        print(f"HTML file saved at {output_path}")
        self.url = f"file://{os.path.abspath(output_path)}"
    # launches the HTML page in default browser
    def launch_page(self):
        webbrowser.open_new_tab(self.url)
