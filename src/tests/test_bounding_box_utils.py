import pytest
import tempfile
import os
from PIL import Image
from src.utils.bounding_box_utils import draw_bounding_boxes, draw_result_bounding_boxes, is_problematic_box, extend_bounding_box_to_edges, InvalidBoundingBoxException

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
    boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
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
    boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    # Call the function to draw bounding boxes
    draw_result_bounding_boxes([str(test_image)], output_path, (120, 120), [[10, 10, 50, 50]], [[60, 60, 90, 90]])
    # Check if the output image exists
    output_img_path = output_path / 'test_input.jpg'
    assert output_img_path.exists()
    # Check if the output image has the expected size
    with Image.open(output_img_path) as img:
      assert img.size == (120, 120)

  def test_is_problematic_box_one_edge(self):
    coords = [0.0, 0.10288684844970702, 0.0860845947265625, 0.88]
    assert is_problematic_box(coords)

  def test_is_problematic_box_two_edges(self):
    coords = [0.0, 0.0, 0.0860845947265625, 0.88]
    assert is_problematic_box(coords)

  def test_is_problematic_box_three_edges(self):
    coords = [0.0, 0.0, 0.0860845947265625, 1.0]
    assert not is_problematic_box(coords)

  def test_is_problematic_box_none(self):
    coords = None
    assert not is_problematic_box(coords)

  def test_is_problematic_box_three_edges_too_big(self):
    coords = [0.0, 0.0616408920288086, 1.0, 1.0]
    assert is_problematic_box(coords)

  def test_extend_bounding_box_to_edges_one_edge_left(self):
    coords = [0.0, 0.10288684844970702, 0.0860845947265625, 0.88]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.0, 0.0860845947265625, 1.0] == result

  def test_extend_bounding_box_to_edges_one_edge_top(self):
    coords = [0.21832839965820314, 0.0, 0.75, 0.05075512886047363]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.0, 1.0, 0.05075512886047363] == result

  def test_extend_bounding_box_to_edges_one_edge_bottom(self):
    coords = [0.2, 0.9267693074544271, 0.913, 1.0]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.9267693074544271, 1.0, 1.0] == result

  def test_extend_bounding_box_to_edges_one_edge_right(self):
    coords = [0.8969069417317709, 0.2, 1.0, 0.888]
    result = extend_bounding_box_to_edges(coords)
    assert [0.8969069417317709, 0.0, 1.0, 1.0] == result

  def test_extend_bounding_box_to_edges_top_left_edge_corner(self):
    coords = [0.0, 0.0, 0.10293843587239583, 0.9715890502929687]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.0, 0.10293843587239583, 1.0] == result

  def test_extend_bounding_box_to_edges_bottom_left_edge_corner(self):
    coords = [0.0, 0.10288684844970702, 0.0860845947265625, 1.0]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.0, 0.0860845947265625, 1.0] == result

  def test_extend_bounding_box_to_edges_top_edge_right_corner(self):
    coords = [0.21832839965820314, 0.0, 1.0, 0.05075512886047363]
    result = extend_bounding_box_to_edges(coords)
    assert [0.0, 0.0, 1.0, 0.05075512886047363] == result

  def test_extend_bounding_box_to_edges_bottom_edge_right_corner(self):
    coords = [0.8969069417317709, 0.3, 1.0, 1.0]
    result = extend_bounding_box_to_edges(coords)
    assert [0.8969069417317709, 0.0, 1.0, 1.0] == result

  def test_extend_bounding_box_to_edges_box_too_big(self):
    coords = [0.0, 0.47508056640625, 0.9, 1.0]
    with pytest.raises(InvalidBoundingBoxException) as e_info:
      result = extend_bounding_box_to_edges(coords)

  def test_extend_bounding_box_to_edges_same_dimensions(self):
    coords = [0.0, 0.0, 0.21832839965820314, 0.21832839965820314]
    with pytest.raises(InvalidBoundingBoxException) as e_info:
      result = extend_bounding_box_to_edges(coords)
