from src.utils.image_normalizer import ImageNormalizer
from src.utils.config import Config
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Normalize an individual image or images from a directory.')
parser.add_argument('src_path', type=Path,
                    help='Path of image or directory to normalize')
parser.add_argument('-o', '--output-path', type=Path, default=Path('./output'),
                    help='Base path to output normalized images. Defaults to "./output"'),
parser.add_argument('-s', '--base-src-path', type=Path,
                    help='Base path which source image paths will be evaluated relative to. If provided, intermediate directories between the base and the file will be recreated in the output directory. Recommended when processing nested subdirectories of images.')
parser.add_argument('-d', '--max-dimension', type=int, default=1333,
                    help='Maximum length of the longest dimension of the image, if exceeded the image will be resized. Default: 1333'),
parser.add_argument('-f', '--force', action="store_true",
                    help='Force generation of normalized versions even if they already exist.'),
parser.add_argument('-n', '--dry-run', action="store_true",
                    help='Perform a dry run, so no normalized files are produced.'),
parser.add_argument('-e', '--extensions', default='tif,tiff,jp2,jpg,jpf,jpx,png',
                    help='List of comma separated file extensions to filter by when operating on a directory. Default: tif,tiff,jp2,jpg,jpf,jpx,png'),


args = parser.parse_args()
extensions = { f".{item.strip(' .')}" for item in args.extensions.split(',') }

print(f'Normalizing images at path: {args.src_path}')
print(f'Outputting to: {args.output_path}')
print(f'For types: {extensions}')

config = Config()
config.max_dimension = args.max_dimension
config.output_base_path = args.output_path
config.src_base_path = args.base_src_path
config.force = args.force

normalizer = ImageNormalizer(config)

if args.src_path.is_dir():
  paths = list(p for p in Path(args.src_path).glob("**/*") if p.suffix in extensions)
else:
  paths = [args.src_path]


total = len(paths)
for idx, path in enumerate(paths):
  print(f"Processing {idx + 1} of {total}: {path}")
  if not args.dry_run:
    normalizer.process(path)