import os
import argparse
from pathlib import Path
import torch

from src.systems.color_bar_segmentation_system import ColorBarSegmentationSystem
from src.utils.common_utils import log

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert a segmentation model to a state dict.')
  parser.add_argument('-m', '--model-path', type=Path, required=True, 
                    help="Path where the model is stored.")
  parser.add_argument('-o', '--output-path', type=Path, required=True, 
                    help="Path the state dict will be stored.")
  args = parser.parse_args()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = ColorBarSegmentationSystem.load_from_checkpoint(args.model_path, map_location = device)
  torch.save(model.state_dict(), args.output_path)
