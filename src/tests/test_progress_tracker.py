import pytest
from pathlib import Path
from src.utils.progress_tracker import ProgressTracker

@pytest.fixture
def tracker(tmp_path):
  log_path = tmp_path / 'progress.log'
  tracker = ProgressTracker(tmp_path / 'progress.log')
  return tracker

class TestProgressTracker:
  PATH_1 = "/path/to/file1.jpg"
  PATH_2 = "/path/to/file2.jpg"

  def test_record_completed_once(self, tracker, tmp_path):
    tracker.record_completed(Path(self.PATH_1))
    progress_list = self.load_progress_list(tmp_path)
    assert progress_list[0] == self.PATH_1
    assert len(progress_list) == 1

  def test_record_completed_multiple(self, tracker, tmp_path):
    tracker.record_completed(Path(self.PATH_1))
    tracker.record_completed(Path(self.PATH_2))
    progress_list = self.load_progress_list(tmp_path)
    assert progress_list[0] == self.PATH_1
    assert progress_list[1] == self.PATH_2
    assert len(progress_list) == 2

  def test_reset_log(self, tracker, tmp_path):
    tracker.record_completed(Path(self.PATH_1))
    tracker.reset_log()
    progress_list = self.load_progress_list(tmp_path)
    assert len(progress_list) == 0

  def test_is_complete(self, tracker):
    tracker.record_completed(Path(self.PATH_1))
    assert tracker.is_complete(self.PATH_1)

  def test_is_complete_false(self, tracker):
    tracker.record_completed(Path(self.PATH_1))
    assert not tracker.is_complete(self.PATH_2)

  def load_progress_list(self, tmp_path):
    with open(tmp_path / 'progress.log') as f:
      return list(line for line in f.read().splitlines())
