# this command-line tool can take two arguments which must be preceded by their respective flags
#   1) -f : (REQUIRED) The CSV file that has four columns: 
#                      original_path, normalized_path, predicted_class, predicted_conf
#   2) -s : (OPTIONAL) The path to the saved html path. If this argument is not specified, 
#                      a 'reports' directory will be created where 'reports.html' will be saved.
#   3) -n : (OPTIONAL) HTTP url to replace the normalized image paths.
#   4) -O : (OPTIONAL) Indicates whether the HTML report should be opened in default browser

from src.utils.classifier_report import ReportGenerator
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-path', type=Path, required=True, 
                    help="Path to csv file that will be used to generate html page.")
parser.add_argument('-s', '--output-path', type=Path, default=False, required=False, 
                    help="Path to output HTML report. Defaults to src/report/report.html")
parser.add_argument('-n', '--normalized-url', type=Path, default=False, required=False,
                    help="HTTP path replace normalized image paths.")
parser.add_argument('-O', '--open', action='store_true', required=False,
                    help="Flag indicates to open the html report in system's default browser.")

args = parser.parse_args()

print(f'CSV path: {args.file_path}')
print(f'Outpath path: {args.output_path}')
print(f'Normalized HTTP URL: {args.normalized_url}')

assert os.path.splitext(args.file_path)[-1].lower() == '.csv'

if args.output_path:
    assert os.path.splitext(args.output_path)[-1].lower() == '.html'


generator = ReportGenerator()

generator.parse_csv(args.file_path)

if args.normalized_url:
    generator.normalize_urls(args.normalized_url)
    
generator.create_html_page()

generator.save_file(args.output_path)


if args.open:
    generator.launch_page()





