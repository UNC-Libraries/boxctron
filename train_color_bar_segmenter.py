import os
import argparse
from pathlib import Path
from pprint import pprint

from src.utils.training_config import TrainingConfig
from src.utils.color_bar_model_trainer import ColorBarModelTrainer
from src.utils.common_utils import log
from src.utils.bounding_box_utils import draw_result_bounding_boxes
import shutil
from src.utils.json_utils import to_json

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
  log('===Writing results===')
  pprint(model_trainer.get_test_results())
  model_trainer.write_test_results()

  # Draw bounding boxes on test images for inspection
  test_preds = model_trainer.get_test_set_predictions()
  output_path = model_trainer.config.log_dir / 'predictions'
  if output_path.exists():
    shutil.rmtree(str(output_path))
  output_path.mkdir()
  resize_dims = (model_trainer.config.max_dimension, model_trainer.config.max_dimension)
  draw_result_bounding_boxes(test_preds['img_paths'], output_path, resize_dims, test_preds['predicted_boxes'], test_preds['target_boxes'])
  # Store info about predicted bounding boxes and scores to a json file for inspection
  predict_json_path = model_trainer.config.log_dir / 'predictions.json'
  if Path.exists(predict_json_path):
    predict_json_path.unlink()
  to_json(test_preds, predict_json_path)