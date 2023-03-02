import pytest
import json
from src.utils.config import Config
from pathlib import Path

class TestConfig:
  def test_construct_default(self):
    config = Config()

    assert config.max_dimension == None
    assert config.min_dimension == None
    assert config.src_base_path == None
    assert config.output_base_path == Path('.')
    assert config.force == False

  def test_construct_with_file(self, tmp_path):
    src_base_path = tmp_path / 'source/base'
    output_base_path = tmp_path / 'output/base'
    config_path = tmp_path / 'config.json'
    config_info = {
      'max_dimension' : 512,
      'min_dimension' : 224,
      'src_base_path' : str(src_base_path),
      'output_base_path' : str(output_base_path),
      'force' : True
    }
    with open(config_path, "w") as config_file:
      json.dump(config_info, config_file)

    config = Config(config_path)

    assert config.max_dimension == 512
    assert config.min_dimension == 224
    assert config.src_base_path == src_base_path
    assert config.output_base_path == output_base_path
    assert config.force == True