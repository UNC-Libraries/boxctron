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
        self.substring = substring
        self.data = False
    
    # parses csv and creates a list of dictionaries  
    def parse_csv(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
             self.data = list(DictReader(f))
        self.add_review_columns()
    
    # normalize given path with given url and substring
    def normalize_to_url(self, original_path, url, substring):
        return re.sub(rf'.+(?={substring})', url, original_path)
    
    # convert image paths with url and substring provided at initialization
    def normalize_item_paths(self):
        for item in self.data:
            item["normalized_path"] = self.normalize_to_url(item['normalized_path'], self.url, self.substring)
    
    def add_review_columns(self):
        for item in self.data:
            item['correct'] = 0
            item['incorrect'] = 0
            
    # if mode = 1 generator constructs statistics about dir tree in files
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
            self.normalize_item_paths()
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
              a.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css")
              a.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css")
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
                .w-15 {
                  width: 15%;
                }
                .w-33 {
                  width: 33%;
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
            with a.main():
              # top button row
              with a.div(id="buttonsTop", klass="container mt-3 ms-2", style="visibility:hidden;"):
                # button to export reviewed rows
                with a.a(href="#", id="reviewLink", style="text-decoration:none;", download="reviewed_data"):
                  a.button(id="reviewExportButton", _t="Export 0 reviewed items", disabled=True, klass="btn btn-success")
                # csv button
                with a.a(href=csv_path, style="text-decoration:none;", download="original_data"):
                  a.button(id='csvButton', _t="Original CSV", klass="btn btn-secondary" )
                if stats:
                # toggle button
                  a.button(id='toggleButton', _t="See Images Report", klass="btn btn-dark")
              # bottom button row
              with a.div(id="buttonsBottom", style="visibility:hidden", klass="container mt-1 ms-2"):
                a.button(id="clearReviewsButton", type="button", klass="btn btn-danger", _t="Clear Selection")
              # loading spinner
              with a.div(id="spinner-container"):
                a.span(id="loading-spinner")
              with a.div(klass="container-fluid w-100 ms-2 me-2"):
                if stats:
                  # stats/agg table
                  with a.div(id='statsTable-container', klass='table-container w-100'):
                    a.table(id='statsTable', klass='display')
                    with a.thead(): pass
                # item-level table
                with a.div(id='imagesTable-container', klass='table-container w-100', style='display:none;' if stats else ''):
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
                $("#buttonsTop").css("visibility", "visible");
              ''')
            # creates image-level datatable
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
                        {{ title: 'Image', data: 'normalized_path', className: "w-33 overflow-auto", render: (d,t,r,m) => '<img class="mw-100" src="'+d+'" style=height:200px; loading="lazy" />'}},
                        {{ title: 'Path', data: 'original_path', className: "w-33", render: (d,t,r,m) => '<p class="overflow-auto text-break mw-100">'+d+'</p>'}},
                        {{ title: 'Class', data: 'predicted_class'}},
                        {{ title: 'Confidence', data: 'predicted_conf', render: $.fn.dataTable.render.number(',', '.', 3, '')}},
                        {{ title: 'Review', data: 'correct', className: "w-15", render: (d,t,r,m) => {{
                          return `<div>
                            <button type="button" class="reviewButton btn btn-outline-success w-50 mb-1" id="correct_${{m.row}}" name="review" value='{{"id": "correct_${{m.row}}", "path": "${{r["original_path"].replace(/'/g, '&apos;')}}", "predicted_class": ${{r["predicted_class"]}}, "review": 1 }}'>Correct</button>
                            <button type="button" class="reviewButton btn btn-outline-danger w-50" id="incorrect_${{m.row}}" name="review" value='{{"id": "incorrect_${{m.row}}", "path": "${{r["original_path"].replace(/'/g, '&apos;')}}", "predicted_class": ${{r["predicted_class"]}}, "review": 0 }}'>Incorrect</button>
                              </div>`
                        }}
                        
                        }}
                    ]
                }});
            ''')
            # if agg stats requested, creates stats table
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
                                    $(td).click( ()=> {{ toggle_button_txt(); filterTable(cData); }});       
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
                    loadReviewChoices();
                  }
                  else if (e.which == 39) {
                    $("table:visible").DataTable().page("next").draw("page");
                    loadReviewChoices();
                  }
              });
            ''')
            # toggles buttons and spinner with timeout
            a('''
              async function toggle_buttons_spinner(secs) {
                  setTimeout(() => {
                    $(document).ready( async () => {
                      await $("#spinner-container").toggle();
                      await $("#buttons").toggle();
                    });
                  }, secs);
                };
            ''')
            # function to toggle button text
            a('''
              let toggle_button_txt = () => {
                if ($("#toggleButton").text() === "See Images Report") {
                  $("#toggleButton").text("See Aggregate Report");
                } else {
                  $("#toggleButton").text("See Images Report");
                }
              };
            ''')
            # function to toggle between tables
            a('''    
              let toggle_tables = async () => {
                await $("#imagesTable-container").toggle();
                await $("#statsTable-container").toggle();
                $("#imagesTable").DataTable().search('').draw().columns.adjust();
                $("#statsTable").DataTable().draw('page').columns.adjust();
                toggle_button_txt();
              };
            ''')
            # updates review export button
            a('''
              let updateReviewButton = () => {
                let reviewItems = localStorage.getItem("reviewItems");
                let len = (reviewItems == null) ? 0 : JSON.parse(reviewItems).length;
                $("#reviewExportButton").text(`Export ${len} reviewed items`)
                if (len == 0) {
                  $("#reviewExportButton").prop("disabled",true);
                  $("#buttonsBottom").css("visibility", "hidden");
                } else {
                  $("#reviewExportButton").prop("disabled",false);
                  $("#buttonsBottom").css("visibility", "visible");
                }
              };
            ''')
            # loads review choices from local storage
            a('''
              let loadReviewChoices = () => {
                if (!localStorage.getItem('reviewItems')) {
                  localStorage.setItem("reviewItems", "[]");  
                } else {
                  let reviewedItems = JSON.parse(localStorage.getItem("reviewItems"))
                  $('.reviewButton').removeClass('active');
                  reviewedItems.forEach(e => {
                    e = JSON.parse(e);
                    if ($(`#${e['id']}`).hasClass('active') === false) {
                      $(`#${e['id']}`).toggleClass('active');
                    }
                  });
                }
              };
            ''')
            # runs update functions
            a('''
              loadReviewChoices();
              updateReviewButton();
            ''')
            # functions to add or remove reviewed items to local storage
            a('''
              let updateReviewItem = (e) => {
                let reviewItems = JSON.parse(localStorage.getItem("reviewItems"));
                if (reviewItems.includes(e.target.value)) {
                  reviewItems = reviewItems.filter(x => JSON.parse(x).id != e.target.id);
                } else {
                  reviewItems.push(e.target.value);
                }
                localStorage.setItem("reviewItems", JSON.stringify(reviewItems));
              };
            ''')
            # updates sibling buttons when a button is pressed
            a('''
              let updateSibling = (e) => {
                let sibID = ''
                if (e.target.nextElementSibling !== null) {
                  sibID = e.target.nextElementSibling.id;
                } else {
                  sibID = e.target.previousElementSibling.id;
                }
                if ($(`#${sibID}`).hasClass("active")){
                  $(`#${sibID}`).toggleClass("active");
                  removeReviewItem(sibID);
                }
              };
            ''')
            # event listener for review button being clicked
            a('''
              $("#imagesTable").on("click", "td .reviewButton", e => {
                e.target.classList.toggle("active");
                updateSibling(e);
                updateReviewItem(e);
                updateReviewButton();
              });
            ''')
            # event listener for clear reviews button being clicked
            a('''
              $("#clearReviewsButton").click(e => {
                $("button.active").toggleClass("active");
                localStorage.setItem("reviewItems", "[]");
                updateReviewButton();
              })
            ''')
            # prompts csv download of reviewed items
            a('''
              $("#reviewExportButton").click( () => {
                let reviewData = JSON.parse(localStorage.getItem("reviewItems"));
                reviewData = reviewData.map(e => {
                    e = JSON.parse(e);
                    e['corrected_class'] = Number(e['predicted_class'] == e['review'])
                    return e
                })
                let csvContent = "data:text/csv;charset=utf-8,"
                csvContent += "path,predicted_class,corrected_class\\n"
                csvContent += reviewData.map(e => `${e["path"]}, ${e["predicted_class"]}, ${e["corrected_class"]}`).join("\\n")
                console.log(csvContent);
                let encodedUri = encodeURI(csvContent);
                $("#reviewLink").attr("href", encodedUri)
                $("#reviewLink").trigger("click");
              })
            ''')
            # function to filter item-level table
            a('''
              let filterTable = async (filter_str) => {
                  await toggle_buttons_spinner("0");
                  await $("#statsTable-container").toggle();
                  await $("#imagesTable-container").toggle();
                  $("#imagesTable").DataTable().search(filter_str).draw().columns.adjust();
                  toggle_button_txt();
                  toggle_buttons_spinner("200");
              }
            ''')
            # sets click event for toggle button
            a('''     
              $("#toggleButton").click( async () => {
                  toggle_buttons_spinner("0");
                  await toggle_tables()
                  await toggle_buttons_spinner("200");
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
