from src.utils.segmentation_utils import background_box

class TestSegmentationUtils:
  def test_background_box_with_origin_vertical_bar(self):
    box = background_box([0, 0, 1, 0.3])
    assert box == (0, 0.3, 1, 1)

  def test_background_box_with_origin_horizontal_bar(self):
    box = background_box([0, 0, 0.3, 1])
    assert box == (0.3, 0, 1, 1)

  def test_background_box_with_offset_vertical_bar(self):
    box = background_box([0.25, 0, 1, 1])
    assert box == (0, 0, 0.25, 1)

  def test_background_box_with_offset_horizontal_bar(self):
    box = background_box([0, 0.4, 1, 1])
    assert box == (0, 0, 1, 0.4)

  def test_background_box_with_no_bar(self):
    box = background_box(None)
    assert box == (0, 0, 1, 1)