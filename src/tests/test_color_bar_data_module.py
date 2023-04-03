from pathlib import Path
import random
from src.datasets import ColorBarDataModule
from src.utils.training_config import TrainingConfig
from src.utils.json_utils import to_json

class TestColorBarDataModule:
  def test_with_simple_data(self, tmp_path):
    random.seed(10)
    config_path = tmp_path / 'config.json'
    to_json({
      'image_list_path' : 'fixtures/mini_file_list.txt',
      'annotations_path' : 'fixtures/mini_annotations.json',
      'base_image_path' : str(Path("fixtures/normalized_images").resolve()),
      'dataset_class' : 'src.datasets.ColorBarClassifyingDataset',
      'max_dimension' : 1333,
      'batch_size' : 2,
      'test_percent': 0.2,
      'val_percent': 0.2
    }, config_path)
    config = TrainingConfig(config_path)

    data_module = ColorBarDataModule(config)

    train_dl = data_module.train_dataloader()
    
    train_list = list(iter(train_dl))
    # Should be two batches in train set (3 items in batches of up to 2)
    assert len(train_list) == 2
    train_batch1_images = train_list[0][0]
    train_batch1_labels = train_list[0][1]
    assert list(train_batch1_images.shape) == [2, 3, 1333, 1333]
    assert list(train_batch1_labels.shape) == [2]
    train_batch2_images = train_list[1][0]
    train_batch2_labels = train_list[1][1]
    assert list(train_batch2_images.shape) == [1, 3, 1333, 1333]
    assert list(train_batch2_labels.shape) == [1]
    # assert len(train_batch2) == 1
    
    # dev should be in a single batch with one entry
    val_dl = data_module.val_dataloader()
    val_list = list(iter(val_dl))
    assert len(val_list) == 1
    val_batch1_images = val_list[0][0]
    val_batch1_labels = val_list[0][1]
    assert list(val_batch1_images.shape) == [1, 3, 1333, 1333]
    assert list(val_batch1_labels.shape) == [1]

    # test should be in a single batch with one entry
    test_dl = data_module.test_dataloader()
    test_list = list(iter(test_dl))
    assert len(test_list) == 1
    test_batch1_images = test_list[0][0]
    test_batch1_labels = test_list[0][1]
    assert list(test_batch1_images.shape) == [1, 3, 1333, 1333]
    assert list(test_batch1_labels.shape) == [1]
