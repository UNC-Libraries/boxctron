import pytest
from PIL import Image
import numpy as np
import random
from src.utils.json_utils import from_json
from src.utils.image_augmentor import ImageAugmentor
from src.utils.augment_config import AugmentConfig
from pathlib import Path

@pytest.fixture
def config(tmp_path):
  conf = AugmentConfig()
  conf.base_image_path = Path('.') / 'fixtures/normalized_images/'
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir()
  conf.annotations_path = Path('.') / 'fixtures/mini_annotations.json'
  conf.annotations_output_path = tmp_path / 'aug_annotations.json'
  conf.file_list_path = Path('.') / 'fixtures/mini_file_list.txt'
  conf.file_list_output_path = tmp_path / 'aug_file_list.txt'
  return conf

class TestImageAugmentor:
  def test_aug_rotation_90_degree(self, config):
    # Seed guarantees correct rotation
    random.seed(42)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, rot_type = subject.aug_rotation(orig_img)
      assert (1076, 1333) == aug_img.size
      assert rot_type == 'r90'
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      assert np.array_equal(np.rot90(orig_data), aug_data)

  def test_aug_rotation_vertical_flip(self, config):
    # Seed guarantees correct rotation
    random.seed(39)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, rot_type = subject.aug_rotation(orig_img)
      width, height = aug_img.size
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      assert (1333, 1076) == aug_img.size
      assert rot_type == 'rfv'
      # augmented image should be the same as the original, flipped vertically
      assert np.array_equal(np.flip(orig_data, 0), aug_data)

  def test_aug_rotation_90_degree_horizontal(self, config):
    # Seed guarantees correct rotation
    random.seed(36)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, rot_type = subject.aug_rotation(orig_img)
      assert (1076, 1333) == aug_img.size
      assert rot_type == 'r90fh'
      # augmented image should be the same as the original, rotated 90 degrees and flipped horizontally
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      assert np.array_equal(np.flip(np.rot90(orig_data), 1), aug_data)

  def test_aug_rotation_horizontal(self, config):
    # Seed guarantees correct rotation
    random.seed(41)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, rot_type = subject.aug_rotation(orig_img)
      assert (1333, 1076) == aug_img.size
      assert rot_type == 'rfh'
      # augmented image should be the same as the original, flipped horizontally
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      assert np.array_equal(np.flip(orig_data, 1), aug_data)

  def test_aug_rotation_small_jitter(self, config):
    # Seed guarantees correct rotation
    random.seed(37)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, rot_type = subject.aug_rotation(orig_img)
      assert (1333, 1076) == aug_img.size
      assert rot_type == 'rsmall'
      # augmented image should be a little different from the originaly
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      assert not np.array_equal(orig_data, aug_data)

  def test_aug_saturation_reduce(self, config):
    # Seed guarantees correct selection
    random.seed(31)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, sat_type = subject.aug_saturation(orig_img)
      assert (1333, 1076) == aug_img.size
      assert sat_type == 's75'
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      # augmented should be same as original, but less saturated
      assert (orig_data < aug_data).all

  def test_aug_saturation_increase(self, config):
    # Seed guarantees correct selection
    random.seed(42)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, sat_type = subject.aug_saturation(orig_img)
      assert (1333, 1076) == aug_img.size
      assert sat_type == 's125'
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      # augmented should be same as original, but more saturated
      assert (orig_data > aug_data).all

  def test_aug_saturation_original(self, config):
    # Seed guarantees correct selection
    random.seed(37)
    subject = ImageAugmentor(config)
    with Image.open('fixtures/normalized_images/ncc/P0004_0483_17486.jpg') as orig_img:
      aug_img, sat_type = subject.aug_saturation(orig_img)
      assert (1333, 1076) == aug_img.size
      assert sat_type == 's100'
      orig_data = np.array(orig_img)
      aug_data = np.array(aug_img)
      # augmented should be same as original
      assert np.array_equal(orig_data, aug_data)

  def test_process(self, config, tmp_path):
    # Seed guarantees correct selection
    random.seed(39)
    subject = ImageAugmentor(config)
    output_path = subject.process(Path('fixtures/normalized_images/ncc/P0004_0483_17486.jpg'))
    subject.persist_annotations()
    with Image.open(output_path) as aug_img:
      assert (1333, 1076) == aug_img.size
      assert output_path.stem == 'P0004_0483_17486_rfv_s100'
      assert output_path.parent == tmp_path / 'output/ncc'
    aug_annos = from_json(config.annotations_output_path)
    assert len(aug_annos) == 13 # 12 original + 1 augmented
    assert aug_annos[12]['image'] == str(output_path)
    assert aug_annos[12]['annotation_id'] == 8
    assert sum(1 for x in config.output_base_path.rglob('*') if x.is_file()) == 1
    # Verify that augmented image was added to the file list output
    with open(config.file_list_output_path) as f:
      lines = f.read().splitlines()
      assert str(output_path) in lines
      assert len(lines) == 13

  def test_process_with_rotation_and_saturation(self, config, tmp_path):
    # Seed guarantees correct selection
    random.seed(42)
    subject = ImageAugmentor(config)
    output_path = subject.process(Path('fixtures/normalized_images/ncc/P0004_0483_17486.jpg'))
    subject.persist_annotations()
    with Image.open(output_path) as aug_img:
      assert (1076, 1333) == aug_img.size
      assert output_path.stem == 'P0004_0483_17486_r90_s75'
      assert output_path.parent == tmp_path / 'output/ncc'
    aug_annos = from_json(config.annotations_output_path)
    assert len(aug_annos) == 13 # 12 original + 1 augmented
    assert aug_annos[12]['image'] == str(output_path)
    assert aug_annos[12]['annotation_id'] == 8
    assert sum(1 for x in config.output_base_path.rglob('*') if x.is_file()) == 1

  def test_process_image_twice(self, config, tmp_path):
    # Seed guarantees correct selection
    random.seed(42)
    subject = ImageAugmentor(config)
    target_path = Path('fixtures/normalized_images/ncc/P0004_0483_17486.jpg')
    output_path = subject.process(target_path)
    subject.persist_annotations()

    output_path2 = subject.process(target_path)
    subject.persist_annotations()
    with Image.open(output_path) as aug_img:
      assert output_path2.stem == 'P0004_0483_17486_r90fh_s100'
      assert output_path2.parent == tmp_path / 'output/ncc'
    aug_annos = from_json(config.annotations_output_path)
    assert len(aug_annos) == 14 # 12 original + 2 augmented
    assert aug_annos[12]['image'] == str(output_path)
    assert aug_annos[12]['annotation_id'] == 8
    assert aug_annos[13]['image'] == str(output_path2)
    assert aug_annos[13]['annotation_id'] == 8
    assert sum(1 for x in config.output_base_path.rglob('*') if x.is_file()) == 2