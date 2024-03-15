# Script which generates an HTML report from a CSV document created by segmenter_predict.py
from src.utils.segmentation_report_service import SegmentationReportService
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-path', type=Path, required=True, 
                    help="Path to csv file that will be used to generate html page.")
parser.add_argument('-d', '--output-path', type=Path, default=False, required=False, 
                    help="Path of the directory to write the report out to. Defaults to src/report")
parser.add_argument('-n', '--norms-relative-path', default='/', required=False,
                    help="File path normalized image paths will be made relative to.")

# command line arguments
args = parser.parse_args()
print(f'CSV path: {args.file_path}')
print(f'Outpath path: {args.output_path}')
print(f'Normalized images path: {args.norms_relative_path}')

# check that input path is a csv file
assert os.path.splitext(args.file_path)[-1].lower() == '.csv'

# Check that the output directory doesn't already exist
assert not args.output_path.exists()

service = SegmentationReportService(args.file_path, args.output_path, args.norms_relative_path)
service.generate()
