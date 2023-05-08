import pytest
from PIL import Image
import numpy as np
import random
from src.utils.image_augmentor import ImageAugmentor
from src.utils.augment_config import AugmentConfig
from pathlib import Path

@pytest.fixture
def config(tmp_path):
  conf = AugmentConfig()
  conf.src_base_path = Path('.') / 'fixtures/normalized_images/'
  conf.output_base_path = tmp_path / 'output'
  conf.output_base_path.mkdir()
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

  def test_process_with_src_base_path(self, config, tmp_path):
    # Seed guarantees correct selection
    random.seed(39)
    subject = ImageAugmentor(config)
    output_path = subject.process(Path('fixtures/normalized_images/ncc/P0004_0483_17486.jpg'))
    with Image.open(output_path) as aug_img:
      assert (1333, 1076) == aug_img.size
      assert output_path.stem == 'P0004_0483_17486_rfv_s100'
      assert output_path.parent == tmp_path / 'output/ncc'

  def test_process_without_src_base_path(self, config, tmp_path):
    # Seed guarantees correct selection
    random.seed(42)
    config.src_base_path = None
    subject = ImageAugmentor(config)
    output_path = subject.process(Path('fixtures/normalized_images/ncc/P0004_0483_17486.jpg'))
    with Image.open(output_path) as aug_img:
      assert (1076, 1333) == aug_img.size
      assert output_path.stem == 'P0004_0483_17486_r90_s75'
      assert output_path.parent == tmp_path / 'output'
