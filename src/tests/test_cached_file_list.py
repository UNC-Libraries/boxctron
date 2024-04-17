from src.utils.cached_file_list import CachedFileList

class TestCachedFileList:
  def test_with_directory(self, tmp_path):
    self.setup_test_dir(tmp_path)

    subject = CachedFileList(self.base_path, { ".jpg", ".tiff" })
    try:
      assert len(subject) == 3
      paths = self.collect_paths(subject)
      assert len(paths) == 3
      assert self.file_path1 in paths
      assert self.file_path2 in paths
      assert self.file_path3 in paths
    finally:
      subject.cache_path.unlink()

  def test_with_directory_from_cache(self, tmp_path):
    self.setup_test_dir(tmp_path)
    subject = CachedFileList(self.base_path, { ".jpg", ".tiff" })
    try:
      assert len(subject) == 3

      # Create an extra file that won't be picked up without a refresh
      file_path5 = self.nested_path1 / "file5.jpg"
      self.write_file(file_path5)
      subject = CachedFileList(self.base_path, { ".jpg", ".tiff" })
      paths = self.collect_paths(subject)
      assert len(subject) == 3
      assert not file_path5 in paths

      # Initialize with refresh so that we pick up the extra item
      subject = CachedFileList(self.base_path, { ".jpg", ".tiff" }, True)
      paths = self.collect_paths(subject)
      assert len(subject) == 4
      assert file_path5 in paths
    finally:
      subject.cache_path.unlink()

  def test_with_file_list(self, tmp_path):
    self.setup_test_dir(tmp_path)
    file_path5 = tmp_path / "file5.tiff"
    self.write_file(file_path5)

    file_list_path = tmp_path / "file_list.txt"
    with open(file_list_path, "w") as file:
      print(str(self.base_path), file=file)
      print(str(file_path5), file=file)

    subject = CachedFileList(file_list_path, { ".jpg", ".tiff" })
    try:
      assert len(subject) == 4
      paths = self.collect_paths(subject)
      assert len(paths) == 4
      assert self.file_path1 in paths
      assert self.file_path2 in paths
      assert self.file_path3 in paths
      assert file_path5 in paths
    finally:
      subject.cache_path.unlink()

  def test_with_file_list_from_cache(self, tmp_path):
    self.setup_test_dir(tmp_path)
    file_path5 = tmp_path / "file5.tiff"
    self.write_file(file_path5)

    file_list_path = tmp_path / "file_list.txt"
    with open(file_list_path, "w") as file:
      print(str(self.base_path), file=file)
      print(str(file_path5), file=file)

    subject = CachedFileList(file_list_path, { ".jpg", ".tiff" })
    try:
      assert len(subject) == 4

      # Create an extra file that won't be picked up without a refresh
      file_path6 = self.nested_path1 / "file6.jpg"
      self.write_file(file_path6)
      subject = CachedFileList(file_list_path, { ".jpg", ".tiff" })
      paths = self.collect_paths(subject)
      assert len(subject) == 4
      assert not file_path6 in paths

      # Initialize with refresh so that we pick up the extra item
      subject = CachedFileList(file_list_path, { ".jpg", ".tiff" }, True)
      paths = self.collect_paths(subject)
      assert len(subject) == 5
      assert file_path6 in paths
    finally:
      subject.cache_path.unlink()

  def setup_test_dir(self, tmp_path):
    self.base_path = tmp_path / "test_base"
    self.nested_path1 = self.base_path / "nested1"
    self.nested_path2 = self.nested_path1 / "nested2"
    self.nested_path2.mkdir(parents=True)
    self.file_path1 = self.base_path / "file1.jpg"
    self.file_path2 = self.base_path / "file2.tiff"
    self.file_path3 = self.nested_path2 / "file3.jpg"
    self.file_path4 = self.nested_path2 / "file4.txt"
    self.write_file(self.file_path1)
    self.write_file(self.file_path2)
    self.write_file(self.file_path3)
    
  def write_file(self, file_path):
    with open(file_path, "w") as text_file:
      text_file.write(str(file_path))

  def collect_paths(self, subject):
    paths = []
    for path in subject:
      paths.append(path)
    return paths
