from pathlib import Path
import random
from src.datasets.color_bar_segmentation_dataset import ColorBarSegmentationDataset
from src.utils.training_config import TrainingConfig
from src.utils.json_utils import to_json
from torch import count_nonzero, all

class TestColorBarSegmentationDataset:
  def test_with_simple_data(self, tmp_path):
    random.seed(10)
    config_path = tmp_path / 'config.json'
    to_json({
      'image_list_path' : 'fixtures/mini_file_list.txt',
      'annotations_path' : 'fixtures/mini_annotations.json',
      'base_image_path' : str(Path("fixtures/normalized_images").resolve()),
      'dataset_class' : 'src.datasets.color_bar_classifying_dataset.ColorBarClassifyingDataset',
      'max_dimension' : 1333
    }, config_path)
    config = TrainingConfig(config_path)

    with open(config.image_list_path) as f:
      image_paths = [Path(p).resolve() for p in f.read().splitlines()]
    dataset = ColorBarSegmentationDataset(config, image_paths)
    assert dataset.__len__() == 13
    assert len(dataset.image_dimensions) == len(dataset)
    assert dataset.image_dimensions[0] == (1333, 964)
    item0, mask0 = dataset.__getitem__(0)
    assert list(item0.shape) == [3, 1333, 1333] 
    assert count_nonzero(mask0) == 0 # Negative example should have an all-zero mask
    assert len(dataset.labels) == 13
    item1, mask1 = dataset.__getitem__(1) 
    assert list(item1.shape) == [3, 1333, 1333] # Check dimensions of cropped image
    assert list(mask1.shape) == [1333, 1333] # Check dimensions of cropped mask
    assert all((mask1 == 0) | (mask1 == 1)) # Verify mask is binary
    assert count_nonzero(mask1) == 365242 # 274 * 1333 pixels are marked as color bars
    item2, mask2 = dataset.__getitem__(6) # Has color bar with margins on all sides after resize
    assert list(item2.shape) == [3, 1333, 1333]
    assert count_nonzero(mask2) == 297868 # 1318 * 226 pixels are marked as color bars
    assert count_nonzero(mask2[0]) == 0 
    assert count_nonzero(mask2[100]) == 1318
    assert count_nonzero(mask2[1332]) == 0