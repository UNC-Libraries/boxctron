from src.utils.segmentation_workflow_service import SegmentationWorkflowService
from src.utils.classifier_config import ClassifierConfig
import argparse
from pathlib import Path
from src.utils.cached_file_list import CachedFileList

parser = argparse.ArgumentParser(description='Use a trained model to segment images in a directory. Normalized versions of the images will be produced if they are not already present')
parser.add_argument('src_path', type=Path,
                    help='Path of image or directory to augment, or file listing paths if the -l parameter is provided')
parser.add_argument('report_path', type=Path,
                    help='Path where the CSV report of results should be written')
parser.add_argument('-c', '--config', type=Path,
                    help='JSON config file for prediction options')
parser.add_argument('-e', '--extensions', default='tif,tiff,jp2,jpg,jpf,jpx,png',
                    help='List of comma separated file extensions to filter by when operating on a directory. Default: tif,tiff,jp2,jpg,jpf,jpx,png')
parser.add_argument('-l', '--file-list', action="store_true",
                    help='If provided, then the src_path will be treated as a text file containing a list of newline separated paths to normalize.')
parser.add_argument('-r', '--restart', action="store_true",
                    help='If provided, then the progress log and CSV report will be discarded and processing will start from the beginning')
parser.add_argument('--refresh', action="store_true",
                    help='If provided, then the list of files to process will be refreshed from disk')
parser.add_argument('-b', '--minimum-bytes', type=int, default=128000,
                    help='Minimum size of files to process, in bytes. Default: 128000')


args = parser.parse_args()
extensions = { f".{item.strip(' .')}" for item in args.extensions.split(',') }

print(f'Segmenting images at path: {args.src_path}')
print(f'For types: {extensions}')

config = ClassifierConfig(path=args.config)

path = None
if args.file_list or args.src_path.is_dir():
  paths = CachedFileList(args.src_path, extensions, args.refresh, minimum_bytes = args.minimum_bytes, cache_path = config.file_list_cache_path)
  print(f'Found {len(paths)} paths for processing')
else:
  paths = [args.src_path]

service = SegmentationWorkflowService(config, Path(args.report_path), args.restart)
service.process(paths)
