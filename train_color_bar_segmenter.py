import os
import argparse
from pathlib import Path
from pprint import pprint

from src.utils.training_config import TrainingConfig
from src.utils.color_bar_model_trainer import ColorBarModelTrainer
from src.utils.common_utils import log
from src.utils.bounding_box_utils import draw_result_bounding_boxes
import shutil

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train color bar segmentation model.')
  parser.add_argument('-c', '--config', dest='config', type=str,
                    default='fixtures/test_seg_config.json',
                    help='Path to training config')
  args = parser.parse_args()

  model_trainer = ColorBarModelTrainer()
  model_trainer.init_system(args.config)
  log('===Training segmentation model===')
  model_trainer.train_model()
  log('===Validation Evaluation===')
  model_trainer.validation_evaluation()
  validation_incorrect = model_trainer.get_validation_incorrect_results_as_csv()
  log(f"Validation Incorrect Results{validation_incorrect}")
  log('===Testing model===')
  model_trainer.offline_test()
  pprint(model_trainer.get_test_results())
  model_trainer.write_test_results()
  test_preds = model_trainer.get_test_set_predictions()
  output_path = model_trainer.config.log_dir / 'predictions'
  shutil.rmtree(str(output_path))
  output_path.mkdir()
  resize_dims = (model_trainer.config.max_dimension, model_trainer.config.max_dimension)
  draw_result_bounding_boxes(test_preds['img_paths'], output_path, resize_dims, test_preds['predicted_boxes'], test_preds['target_boxes'])
  log(f'Test Incorrect Results\n{model_trainer.get_test_incorrect_results_as_csv()}')