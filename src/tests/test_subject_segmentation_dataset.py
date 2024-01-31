from pathlib import Path
import pytest
import random
from src.datasets.subject_segmentation_dataset import SubjectSegmentationDataset
from src.utils.training_config import TrainingConfig
from src.utils.json_utils import to_json
from torch import count_nonzero, all

@pytest.fixture
def config(tmp_path):
  config_path = tmp_path / 'config.json'
  to_json({
    'image_list_path' : 'fixtures/mini_file_list.txt',
    'annotations_path' : 'fixtures/mini_annotations.json',
    'base_image_path' : str(Path("fixtures/normalized_images").resolve()),
    'dataset_class' : 'src.datasets.subject_segmentation_dataset.SubjectSegmentationDataset',
    'max_dimension' : 1333
  }, config_path)
  return TrainingConfig(config_path)

@pytest.fixture
def image_paths(config):
  with open(config.image_list_path) as f:
    return [Path(p).resolve() for p in f.read().splitlines()]

class TestSubjectSegmentationDataset:

  def test_with_simple_data(self, config, image_paths):
    random.seed(10)
    dataset = SubjectSegmentationDataset(config, image_paths)
    assert dataset.__len__() == 14
    # assert len(dataset.image_dimensions) == len(dataset)
    # assert dataset.image_dimensions[0] == (1333, 964)
    item0, target0 = dataset.__getitem__(0)
    # mask0 = target0['masks'][0]
    assert list(item0.shape) == [3, 1333, 1333] 
    # assert count_nonzero(mask0) == 0 # Negative example should have an all-zero mask
    assert len(dataset.labels) == 14
     # Negative example, entire image is subject
    assert target0['boxes'].data.tolist() == [[0.0, 0.0, 1333.0, 1333.0]]
    # assert target0['boxes'].data.tolist() == []
    # assert target0['labels'].data.tolist() == [0]
    assert target0['labels'].data.tolist() == [1]
    item1, target1 = dataset.__getitem__(1)
    # mask1 = target1['masks'][0]
    assert list(item1.shape) == [3, 1333, 1333] # Check dimensions of cropped image
    # assert list(mask1.shape) == [1333, 1333] # Check dimensions of cropped mask
    # assert all((mask1 == 0) | (mask1 == 1)) # Verify mask is binary
    # assert count_nonzero(mask1) == 365242 # 274 * 1333 pixels are marked as color bars
    print(f"Image1 {target1['img_path']}")
    # assert target1['boxes'].data.tolist() == [[0.0, 0.0, 1333.0, 1029.0], [0.0, 1029.0, 1333.0, 1333.0]]
    assert target1['boxes'].data.tolist() == [[0.0, 0.0, 1333.0, 1025.0]]
    # assert target1['labels'].data.tolist() == [0, 1]
    assert target1['labels'].data.tolist() == [1]
    item2, target2 = dataset.__getitem__(6) # Has color bar with margins on all sides before rounding
    # mask2 = target2['masks'][0]
    assert list(item2.shape) == [3, 1333, 1333]
    # Color bar label gets rounded to nearest edges since they are within threshold
    # assert count_nonzero(mask2) == 301258 # 1333 * 226 pixels are marked as color bars
    # assert count_nonzero(mask2[0]) == 0 
    # assert count_nonzero(mask2[100]) == 1333
    # assert count_nonzero(mask2[1332]) == 0
    print(f"Image2 {target2['img_path']}")
    # assert target2['boxes'].data.tolist() == [[0.0, 236.0, 1333.0, 1333.0], [0.0, 0.0, 1333.0, 236.0]]
    # assert target2['labels'].data.tolist() == [0, 1]
    assert target2['boxes'].data.tolist() == [[0.0, 232.0, 1333.0, 1333.0]]
    assert target2['labels'].data.tolist() == [1]
    # Uncomment to generate images with masks applied
    # dataset.visualize_tensor(item2, mask2)
    # dataset.visualize_tensor(item0, mask0)
    # Item has a really small color bar, smaller than the rounding threshold, so don't round
    item3, target3 = dataset.__getitem__(13)
    # assert target3['boxes'].data.tolist() == [[0.0, 0.0, 1333.0, 1319.0], [0.0, 1319.0, 1333.0, 1333.0]]
    # assert target3['labels'].data.tolist() == [0, 1]
    assert target3['boxes'].data.tolist() == [[0.0, 0.0, 1333.0, 1319.0]]
    assert target3['labels'].data.tolist() == [1]