# this command-line tool can take two arguments which must be preceded by their respective flags
#   1) -f : (REQUIRED) The CSV file that has four columns: 
#                      original_path, normalized_path, predicted_class, predicted_conf
#   2) -s : (OPTIONAL) The path to the saved html path. If this argument is not specified, 
#                      a 'reports' directory will be created where 'reports.html' will be saved.
#   3) -x : (OPTIONAL) Substring that indicates the start of the normalized image path that will follow the normalized-url.
#   4) -n : (OPTIONAL) HTTP url which the normalized image paths will be made relative to.
#   5) -O : (OPTIONAL) Indicates whether the HTML report should be opened in default browser
#   6) -A : (OPTIONAL) Indicates whether aggregate data table will be created
from src.utils.classifier_report import DataParser, ReportGenerator
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-path', type=Path, required=True, 
                    help="Path to csv file that will be used to generate html page.")
parser.add_argument('-s', '--output-path', type=Path, default=False, required=False, 
                    help="Path to output HTML report. Defaults to src/report/report.html")
parser.add_argument('-x', '--substring', default="/shared/", required=False,
                    help="Substring that indicates the start of the normalized image path that will follow the normalized-url.")
parser.add_argument('-n', '--normalized-url', default=False, required=False,
                    help="URL which normalized image paths will be made relative to.")
parser.add_argument('-O', '--open', action='store_true', required=False,
                    help="Flag indicates to open the html report in system's default browser.")
parser.add_argument('-A', '--agg-data', action='store_true', required=False,
                    help="Parser will provide additional table with aggregate data.")

args = parser.parse_args()

print(f'CSV path: {args.file_path}')
print(f'Outpath path: {args.output_path}')
print(f'Normalized HTTP URL: {args.normalized_url}')

assert os.path.splitext(args.file_path)[-1].lower() == '.csv'

if args.output_path:
    assert os.path.splitext(args.output_path)[-1].lower() == '.html'

parser = DataParser(args.file_path, args.normalized_url, args.substring)
generator = ReportGenerator()

data = parser.get_data()
stats = False
if args.agg_data:
    stats = parser.get_stats()
generator.create_html_page(data, stats)
generator.save_file(args.output_path)

if args.open:
    generator.launch_page()
