from src.datasets.color_bar_dataset import ColorBarDataset
from pathlib import Path

# Dataset with single class for labeling images as having a color bar or not
class ColorBarClassifyingDataset(ColorBarDataset):
  def __init__(self, config, image_paths, split = 'train'):
    super().__init__(config, image_paths, split)

  # Loads annotation data into self.labels in the same order they paths are listed in image_paths
  def load_labels(self, path_to_labels):
    count_by_label = [0, 0]
    for index, image_path in enumerate(self.image_paths):
      image_labels = path_to_labels[str(image_path)]
      has_cb = 0
      for label in image_labels:
        if 'color_bar' in label['rectanglelabels']:
          has_cb = 1
          break
      self.labels.append(has_cb)
      count_by_label[has_cb] = count_by_label[has_cb] + 1
    print(f'Dataset {self.split} contains counts for labels: {count_by_label}')
