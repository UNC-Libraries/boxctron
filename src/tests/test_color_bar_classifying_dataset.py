from pathlib import Path
import random
from src.datasets.color_bar_classifying_dataset import ColorBarClassifyingDataset
from src.utils.training_config import TrainingConfig
from src.utils.json_utils import to_json

class TestColorBarClassifyingDataset:
  def test_with_simple_data(self, tmp_path):
    random.seed(10)
    config_path = tmp_path / 'config.json'
    to_json({
      'image_list_path' : 'fixtures/mini_file_list.txt',
      'annotations_path' : 'fixtures/mini_annotations.json',
      'base_image_path' : str(Path("fixtures/normalized_images").resolve()),
      'dataset_class' : 'color_bar_classifying_dataset'
    }, config_path)
    config = TrainingConfig(config_path)

    dataset = ColorBarClassifyingDataset(config)
    assert dataset.__len__() == 5
    item0 = dataset.__getitem__(0)
    assert list(item0[0].shape) == [3, 964, 1333]
    assert item0[1] == 0
    item1 = dataset.__getitem__(1)
    assert list(item1[0].shape) == [3, 1205, 1333]
    assert item1[1] == 1
    assert dataset.__getitem__(2)[1] == 1
    assert dataset.__getitem__(3)[1] == 0
    assert dataset.__getitem__(4)[1] == 1
