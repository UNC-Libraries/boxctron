from pathlib import Path
import random
from src.datasets.color_bar_classifying_dataset import ColorBarClassifyingDataset

class TestColorBarClassifyingDataset:
  def test_with_simple_data(self):
    random.seed(10)
    image_paths = [
      Path("fixtures/normalized_images/ncc/Cm912_1945u1_sheet1.jpg").resolve(),
      Path("fixtures/normalized_images/ncc/P0004_0483_0001_verso.jpg").resolve(),
      Path("fixtures/normalized_images/ncc/P0004_0483_17486.jpg").resolve(),
      Path("fixtures/normalized_images/ncc/G3902-F3-1981_U5_front.jpg").resolve(),
      Path("fixtures/normalized_images/rbc/11029-z_pa0001_0017.jpg").resolve()]

    dataset = ColorBarClassifyingDataset(image_paths, Path("fixtures/mini_annotations.json"), Path("fixtures/normalized_images"))
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
