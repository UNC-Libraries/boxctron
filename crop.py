from src.utils.cropping_workflow_service import CroppingWorkflowService
from src.utils.classifier_config import ClassifierConfig
import argparse
from pathlib import Path
from datetime import datetime
from src.utils.common_utils import log

parser = argparse.ArgumentParser(description='Using a prediction CSV from segmenter_predict.py, crop original images and save them to an output directory')
parser.add_argument('csv_path', type=Path,
                    help='Path to the prediction CSV file')
parser.add_argument('output_path', type=Path,
                    help='Base path where the cropped images will be written to.')
parser.add_argument('-e', '--exclusions', type=Path,
                    help='If provided, CSV file will be loaded and any file paths in the first column will be skipped during cropping')
parser.add_argument('-c', '--config', type=Path,
                    help='JSON config file for prediction options')

args = parser.parse_args()

log(f'Cropping images from CSV: {args.csv_path}')
log(f'Writing cropped images to: {args.output_path}')

config = ClassifierConfig(path=args.config)

service = CroppingWorkflowService(args.csv_path, args.output_path, config, exclusions_path = args.exclusions)

cropped_paths = service.process()
cropped_report = args.output_path / f"cropped{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
with open(cropped_report, 'w') as file:
  for cropped in cropped_paths:
    file.write(f'{str(cropped)}\n')

log(f'Wrote list of cropped files to {cropped_report.resolve()}')
