from pathlib import Path
import json

# Utility for tracking what items have been processed already, to support resumption
class ProgressTracker:
  def __init__(self, log_path):
    self.log_path = log_path
    self.completed = None

  def reset_log(self):
    with open(self.log_path, "a") as log_file:
      log_file.truncate(0)

  def completed_set(self):
    if self.completed == None:
      if self.log_path.is_file():
        with open(self.log_path) as f:
          self.completed = set(line for line in f.read().splitlines())
      else:
        self.completed = set([])
    return self.completed

  # Check if the given path has been completed in a previous run
  def is_complete(self, path):
    return str(path) in self.completed_set()

  # Record a completed path to the log
  def record_completed(self, path):
    with open(self.log_path, "a") as log_file:
      print(str(path), file=log_file)