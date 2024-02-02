import os
import argparse
from pathlib import Path
from pprint import pprint

from src.utils.training_config import TrainingConfig
from src.utils.color_bar_model_trainer import ColorBarModelTrainer
from src.utils.common_utils import log

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train color bar classifier.')
  parser.add_argument('-c', '--config', dest='config', type=str,
                    default='fixtures/test_config.json',
                    help='Path to training config')
  args = parser.parse_args()

  train_classifier = ColorBarModelTrainer()
  train_classifier.init_system(args.config)
  log('Training model')
  train_classifier.train_model()
  log('Validation Evaluation')
  train_classifier.validation_evaluation()
  validation_incorrect = train_classifier.get_validation_incorrect_results_as_csv()
  log(f"Validation Incorrect Results{validation_incorrect}")
  log('Testing model')
  train_classifier.offline_test()
  pprint(train_classifier.get_test_results())
  train_classifier.write_test_results()
  log(f'Test Incorrect Results\n{train_classifier.get_test_incorrect_results_as_csv()}')