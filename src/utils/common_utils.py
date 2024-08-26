from datetime import datetime
from pathlib import Path

_all__ = ["log", "rebase_path"]

def log(message):
  print(f'{datetime.now().isoformat()} {message}')

# Change a Path object so that the portion of it that is relative to the current_base is now
# relative to the new base, and has a suffix added.
# For example, given:
#   path = Path(/my/special/file.tif)
#   new_base = Path(/a/new/base)
#   suffix = '.jpg'
# Would produce Path(/a/new/base/my/special/file.tif.jpg)
def rebase_path(new_base, path, suffix):
  rel_path = str(path)
  if rel_path.startswith('/'):
    rel_path = rel_path[1:]
  return new_base / (rel_path + suffix)
