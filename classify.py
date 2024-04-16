from src.utils.classifier_workflow_service import ClassifierWorkflowService
from src.utils.classifier_config import ClassifierConfig
import argparse
from pathlib import Path
from src.utils.common_utils import add_expanded_dir, recursive_paths_from_file_list

parser = argparse.ArgumentParser(description='Use a trained model to classify images in a directory. Normalized versions of the images will be produced if they are not already present')
parser.add_argument('src_path', type=Path,
                    help='Path of image or directory to augment, or file listing paths if the -l parameter is provided')
parser.add_argument('report_path', type=Path,
                    help='Path where the CSV report of results should be written')
parser.add_argument('-c', '--config', type=Path,
                    help='If provided, config will be loaded from this file instead of commandline options')
parser.add_argument('-e', '--extensions', default='tif,tiff,jp2,jpg,jpf,jpx,png',
                    help='List of comma separated file extensions to filter by when operating on a directory. Default: tif,tiff,jp2,jpg,jpf,jpx,png'),
parser.add_argument('-l', '--file-list', action="store_true",
                    help='If provided, then the src_path will be treated as a text file containing a list of newline separated paths to normalize.'),
parser.add_argument('-r', '--restart', action="store_true",
                    help='If provided, then the progress log and CSV report will be discarded and processing will start from the beginning'),


args = parser.parse_args()
extensions = { f".{item.strip(' .')}" for item in args.extensions.split(',') }

print(f'Classifying images at path: {args.src_path}')
print(f'For types: {extensions}')

config = ClassifierConfig(path=args.config)

if args.file_list:
  paths = recursive_paths_from_file_list(args.src_path)
elif args.src_path.is_dir():
  paths = add_expanded_dir(args.src_path, [])
else:
  paths = [args.src_path]

service = ClassifierWorkflowService(config, Path(args.report_path), args.restart)
service.process(paths)
