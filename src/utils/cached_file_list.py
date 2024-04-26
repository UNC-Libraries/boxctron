from pathlib import Path

class CachedFileList(list):
  """
  List implementation which pulls entries in the list from a cache file on disk.
  The cache will be populated on initialization if the cache does not already exist or
  it has been requested to be refreshed.
  Entries for the cache are taken from the provided file_path. If the file_path is
  a directory, then all valid files in the directory will be added. If file_path is
  a file, then each line in the file will be treated as a path to be added, where any
  directories in the file will be expanded.
  """
  def __init__(self, file_path, extensions, refresh = False, minimum_bytes = 4000):
    super().__init__()
    self.file_path = file_path
    self.extensions = extensions
    self.cache_path = Path.cwd() / (file_path.stem + "-cache.txt")
    self.minimum_bytes = minimum_bytes
    if not self.cache_path.exists() or refresh:
      self.populate_cache()
    with open(self.cache_path, "r") as file:
      self.total_entries = sum(1 for line in file)

  def populate_cache(self):
    if self.cache_path.exists():
      self.cache_path.unlink()
    with open(self.cache_path, "w") as file:
      self.file = file
      if self.file_path.is_dir():
        # expand dir
        self.add_expanded_dir(self.file_path)
      else:
        # parse as a file list
        self.recursive_paths_from_file_list()

  def add_expanded_dir(self, dir_path):
    for path in Path(dir_path).glob("**/*"):
      if path.suffix in self.extensions:
        self.add_file_path(path)

  def recursive_paths_from_file_list(self):
    with open(self.file_path) as f:
      for line in f.read().splitlines():
        path = Path(line.strip())
        if path.is_dir():
          print(f"Expanding path {path}")
          self.add_expanded_dir(path)
        else:
          self.add_file_path(path)

  # Adds the given file path to the cache file if it has an acceptable extension and is above the min size
  def add_file_path(self, path):
    if path.suffix in self.extensions:
      file_size = path.stat().st_size
      if file_size >= self.minimum_bytes:
        print(str(path), file=self.file)
      else:
        print(f"Skipping small file {path}")

  def __iter__(self):
    self.file = open(self.cache_path, "r")
    return self

  def __next__(self):
    line = self.file.readline().strip()
    if line == "":
      self.file.close()
      raise StopIteration
    return Path(line)

  def close(self):
    if self.file:
      self.file.close()

  def __len__(self):
    return self.total_entries
