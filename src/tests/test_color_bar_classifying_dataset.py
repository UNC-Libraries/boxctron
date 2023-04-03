from pathlib import Path
import random
from src.datasets import ColorBarClassifyingDataset
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
      'dataset_class' : 'src.datasets.ColorBarClassifyingDataset',
      'max_dimension' : 1333
    }, config_path)
    config = TrainingConfig(config_path)

    with open(config.image_list_path) as f:
      image_paths = [Path(p).resolve() for p in f.read().splitlines()]
    dataset = ColorBarClassifyingDataset(config, image_paths)
    assert dataset.__len__() == 5
    item0 = dataset.__getitem__(0)
    assert list(item0[0].shape) == [3, 1333, 1333]
    assert item0[1] == 0
    item1 = dataset.__getitem__(1)
    assert list(item1[0].shape) == [3, 1333, 1333]
    assert item1[1] == 1
    assert dataset.__getitem__(2)[1] == 1
    assert dataset.__getitem__(3)[1] == 0
    assert dataset.__getitem__(4)[1] == 1
