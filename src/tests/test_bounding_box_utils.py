import pytest
import tempfile
import os
from torch import tensor
from PIL import Image
from src.utils.bounding_box_utils import draw_bounding_boxes, draw_result_bounding_boxes

@pytest.fixture
def test_image(tmp_path):
    # Create a temporary input image
    input_img_path = tmp_path / 'test_input.jpg'
    print(f'Path {tmp_path} {os.path.exists(tmp_path)}')
    img = Image.new('RGB', (100, 100), color='white')
    img.save(input_img_path)
    return input_img_path

class TestBoundingBoxUtils:
  def test_draw_bounding_boxes(self, test_image, tmp_path):
    output_img_path = tmp_path / 'output.jpg'
    # Define some test bounding boxes
    boxes = [tensor([10, 10, 50, 50]), tensor([60, 60, 90, 90])]
    # Call the function to draw bounding boxes
    draw_bounding_boxes(test_image, output_img_path, (120, 120), boxes)
    # Check if the output image exists
    assert os.path.exists(output_img_path)
    # Check if the output image has the expected size
    with Image.open(output_img_path) as img:
      assert img.size == (120, 120)

  def test_draw_result_bounding_boxes(self, test_image, tmp_path):
    output_path = tmp_path / 'outputs'
    output_path.mkdir()
    # Define some test bounding boxes
    boxes = [tensor([10, 10, 50, 50]), tensor([60, 60, 90, 90])]
    # Call the function to draw bounding boxes
    draw_result_bounding_boxes([str(test_image)], output_path, (120, 120), [tensor([10, 10, 50, 50])], [tensor([60, 60, 90, 90])])
    # Check if the output image exists
    output_img_path = output_path / 'test_input.jpg'
    assert output_img_path.exists()
    # Check if the output image has the expected size
    with Image.open(output_img_path) as img:
      assert img.size == (120, 120)
