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
        self.stats = defaultdict(lambda: {'path': '', 'count': 0, 'count_CB': 0, 'percent_CB': 0, 'has_CB': "False", 'avg_conf_CB': 0}) 
        for item in self.data:
            folders = item['original_path'].lstrip('/').split('/')
            current_path = ''
            for folder in folders[:-1]:
                current_path += f'/{folder}'
                self.stats[current_path]['path'] = current_path
                self.stats[current_path]['count'] +=1
                if item['predicted_class'] == '1':
                    self.stats[current_path]['count_CB'] += 1
                    self.stats[current_path]['has_CB'] = "True"
                    self.stats[current_path]['avg_conf_CB'] += float(item['predicted_conf'])
        
        for stats in self.stats.values():
            if stats['has_CB'] == "True":
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
            self.get_data()
        self.create_stats()
        return list(self.stats.values())


class ReportGenerator:
    
    # creates the HTML page
    def create_html_page(self, data, csv_path, stats=False):
        a = Airium()
        a('<!DOCTYPE html>')
        with a.html(lang='en'):
            # header info
            with a.head():
                a.meta(charset='UTF-8')
                a.meta(name="viewport", content="width=device-width, initial-scale=1.0")
                a.title(_t="Model report")
                a.title(_t='Classifier Results')
                # CSS CDN
                a.link(rel="stylesheet", href="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-1.13.6/r-2.5.0/sp-2.2.0/sl-1.7.0/datatables.min.css")
                # styling for spinner
                with a.style():
                    a('''
                        #spinner-container {
                            height: 100%;
                            display: flex;
                            justify-content: center;
                            align-items: center;  
                        }
                        #loading-spinner {
                            width: 100px;
                            height: 100px;
                            border: 4px #ddd solid;
                            border-top: 4px #2e93e6 solid;
                            border-radius: 50%;
                            animation: sp-anime 0.8s infinite linear;
                        }
                        @keyframes sp-anime {
                            100% { 
                                transform: rotate(360deg); 
                            }
                        }
                      ''')
            with a.body():
                # csv button
                with a.div(id="buttons", style="display:flex; visibility:hidden"):
                    with a.a(href=csv_path, download="original_data"):
                        a.button(id='csvButton', _t="Original CSV", style="height:30px;padding:5px 8px; background-color:#2ea44f; color:#fff; margin-right:10px; border-style:none; border-radius:4px; cursor:pointer;")
                    if stats:
                    # toggle button
                        a.button(id='toggleButton', _t="See Images Report", style="height:30px;padding:5px 8x; background-color: #678aaa; color: #fff; border-style: none; border-radius:4px; cursor:pointer;")
                # loading spinner
                with a.div(id="spinner-container"):
                    a.span(id="loading-spinner")
                if stats:
                    # toggle button
                    # a.button(id='toggleButton', _t="See Images Report")
                    # stats/agg table
                    with a.div(id='statsTable-container', klass='table-container'):
                        a.table(id='statsTable', klass='display')
                        with a.thead(): pass
                # item-level table
                with a.div(id='imagesTable-container', klass='table-container', style='display:none;' if stats else ''):
                    a.table(id='imagesTable', klass='display')
                    with a.thead(): pass
                

                # JQuery Core and UI CDNs
                a.script(src="https://code.jquery.com/jquery-3.7.0.js", integrity="sha256-JlqSTELeR4TLqP0OG9dxM7yDPqX1ox/HfgiSLBj8+kM=", crossorigin="anonymous")
                a.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.js", integrity="sha256-xLD7nhI62fcsEZK2/v8LsBcb4lG7dgULkuXoXB/j91c=", crossorigin="anonymous")
                # Datables CDN
                a.script(src="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-1.13.6/r-2.5.0/sp-2.2.0/sl-1.7.0/datatables.min.js")
                
                # Javascript creating the table the 'Path' column accounts for the csv 
                with a.script():
                    a("$(document).ready( () => {")
                    # toggles off spinner and displays buttons
                    a('''
                       $("#spinner-container").toggle();
                       $("#buttons").css("visibility", "visible");
                      ''')
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
                            searchPanes: {{viewTotal: true, columns: [2],layout: 'columns-4', initCollapsed: true}},
                            dom: 'Plfrtip',
                            columns: [
                                {{ title: 'Image', data: 'normalized_path', width: "25%", render: (d,t,r,m) => '<img src="'+d+'" style=height:200px; loading="lazy" />'}},
                                {{ title: 'Path', data: 'original_path'}},
                                {{ title: 'Class', data: 'predicted_class'}},
                                {{ title: 'Confidence', data: 'predicted_conf', render: $.fn.dataTable.render.number(',', '.', 3, '')}}
                            ]
                        }});
                    ''')
                    if stats:
                        a(f'''
                            $("#statsTable").DataTable({{
                                data: {stats},
                                pageLength: 50,
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
                                            $(td).click( ()=> {{ toggle_button(); filterTable(cData); }});       
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
                        # enable keystrokes for navigation
                        a('''
                           $("body").on('keydown', (e) => {
                               if (e.which == 37) {
                                $("table:visible").DataTable().page("previous").draw('page');
                               }
                               else if (e.which == 39) {
                                $("table:visible").DataTable().page("next").draw("page");
                               }
                           });
                           ''')
                        # function to toggle button text
                        a('''
                            let toggle_button = () => {{
                                if ($("#toggleButton").text() === "See Images Report") {{
                                    $("#toggleButton").text("See Aggregate Report");
                                    }} else {{
                                        $("#toggleButton").text("See Images Report");
                                    }}
                            }};
                          ''')
                        # function to toggle between tables
                        a('''    
                            let toggle_tables = () => {{
                                $("#imagesTable-container").toggle();
                                $("#imagesTable").DataTable().search('').draw().columns.adjust();
                                $("#statsTable-container").toggle();
                                $("#statsTable").DataTable().draw('page').columns.adjust();
                                toggle_button();
                                }};
                           ''')
                        # function to filter item-level table
                        a('''
                            let filterTable = (filter_str) => {{
                                $("#statsTable-container").toggle();
                                $("#imagesTable-container").toggle();
                                $("#imagesTable").DataTable().search(filter_str).draw().columns.adjust();
                                }}
                            ''')
                         # sets click event for toggle button
                        a('''     
                            $("#toggleButton").click( () => {
                                $("#spinner-container").toggle();
                                $("#buttons").toggle();
                                toggle_tables();
                                $(document).ready(() => {
                                  $("#buttons").toggle();
                                  $("#spinner-container").toggle()
                                });
                            });
                            ''')
                    # closing tags for $(document).ready    
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
