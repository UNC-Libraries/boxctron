from datetime import datetime
from pathlib import Path

_all__ = ["log", "add_expanded_dir", "recursive_paths_from_file_list"]

def log(message):
  print(f'{datetime.now().isoformat()} {message}')

def add_expanded_dir(dir_path, paths, extensions):
  for p in Path(dir_path).glob("**/*"):
    if p.suffix in extensions:
      paths.append(p)
  return paths

def recursive_paths_from_file_list(file_list_path, extensions):
  with open(file_list_path) as f:
    paths = []
    for line in f.read().splitlines():
      path = Path(line.strip())
      if path.is_dir():
        add_expanded_dir(path, paths)
      else:
        paths.append(path)
    return paths