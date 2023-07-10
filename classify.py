from src.utils.classifier_workflow_service import ClassifierWorkflowService
from src.utils.classifier_config import ClassifierConfig
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Use a trained model to classify images in a directory. Normalized versions of the images will be produced if they are not already present')
parser.add_argument('src_path', type=Path,
                    help='Path of image or directory to augment, or file listing paths if the -l parameter is provided')
parser.add_argument('-c', '--config', type=Path,
                    help='If provided, config will be loaded from this file instead of commandline options')
parser.add_argument('-e', '--extensions', default='tif,tiff,jp2,jpg,jpf,jpx,png',
                    help='List of comma separated file extensions to filter by when operating on a directory. Default: tif,tiff,jp2,jpg,jpf,jpx,png'),
parser.add_argument('-l', '--file-list', action="store_true",
                    help='If provided, then the src_path will be treated as a text file containing a list of newline separated paths to normalize.'),


args = parser.parse_args()
extensions = { f".{item.strip(' .')}" for item in args.extensions.split(',') }

print(f'Classifying images at path: {args.src_path}')
print(f'For types: {extensions}')

config = ClassifierConfig(path=args.config)

if args.file_list:
  with open(args.src_path) as f:
    paths = list(Path(line) for line in f.read().splitlines())
elif args.src_path.is_dir():
  paths = list(p for p in Path(args.src_path).glob("**/*") if p.suffix in extensions)
else:
  paths = [args.src_path]

service = ClassifierWorkflowService(config)
service.process(paths)
