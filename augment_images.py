from src.utils.image_augmentor import ImageAugmentor
from src.utils import AugmentConfig
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Augment images in a directory.')
parser.add_argument('src_path', type=Path,
                    help='Path of image or directory to augment, or file listing paths if the -l parameter is provided')
parser.add_argument('-l', '--file-list', action="store_true",
                    help='If provided, then the src_path will be treated as a text file containing a list of newline separated paths to augment.'),
parser.add_argument('-o', '--output-path', type=Path, default=Path('./output'),
                    help='Base path to output augmented images. Defaults to "./output"'),
parser.add_argument('-s', '--base-image-path', type=Path,
                    help='Base path which source image paths will be evaluated relative to. If provided, intermediate directories between the base and the file will be recreated in the output directory. Recommended when processing nested subdirectories of images.')
parser.add_argument('-a', '--annotation-path', type=Path,
                    help='Path to the annotation file associated with the file being augmented.')
parser.add_argument('-A', '--annotation-output-path', type=Path,
                    help='Path to write the updated annotation file to.')
parser.add_argument('-e', '--extensions', default='jpg',
                    help='List of comma separated file extensions to filter by when operating on a directory. Default: jpg')
parser.add_argument('-c', '--config', type=Path,
                    help='If provided, config will be loaded from this file instead of commandline options')


args = parser.parse_args()
extensions = { f".{item.strip(' .')}" for item in args.extensions.split(',') }

print(f'Augmenting images at path: {args.src_path}')
print(f'Outputting to: {args.output_path}')
print(f'For types: {extensions}')

if args.config:
  config = AugmentConfig(path=args.config)
else:
  config = AugmentConfig()
  config.output_base_path = args.output_path
  config.base_image_path = args.base_image_path
  config.annotations_path = args.annotations_path
  config.annotations_output_path = args.annotations_output_path

augmentor = ImageAugmentor(config)
augmentor.persist_annotations

if args.file_list:
  with open(args.src_path) as f:
    paths = list(Path(line) for line in f.read().splitlines())
elif args.src_path.is_dir():
  paths = list(p for p in Path(args.src_path).glob("**/*") if p.suffix in extensions)
else:
  paths = [args.src_path]

total = len(paths)
for idx, path in enumerate(paths):
  print(f"Processing {idx + 1} of {total}: {path}")
  if not args.dry_run:
    augmentor.process(path)

print(f"Updating annotations file to {config.annotations_output_path}")
augmentor.persist_annotations()

