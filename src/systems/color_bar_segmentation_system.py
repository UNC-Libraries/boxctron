import pytorch_lightning as pl
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import BinaryConfusionMatrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import pandas as pd
import io
from PIL import Image
from src.utils.resnet_utils import resnet_foundation_model
from src.utils.iou_utils import evaluate_iou, evaluate_giou
from torchvision.models.detection.faster_rcnn import (fasterrcnn_resnet50_fpn, FasterRCNN, FastRCNNPredictor,)
import pdb
from src.utils.common_utils import log

# System for training a model to classify images as either containing a color bar or not.
# It uses a resnet model as its foundation for transfer learning, then trains on top of that
# for our specific task.
class ColorBarSegmentationSystem(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.validation_step_iou = []
    self.validation_step_giou = []
    self.validation_step_loss = []
    self.test_step_iou = []
    self.test_step_giou = []
    self.test_step_loss = []
    self.test_step_predicted_boxes = []
    self.test_step_predicted_scores = []
    self.test_step_target_boxes = []
    self.test_step_image_paths = []

    self.model = fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=3)
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # We will overwrite this once we run `test()`
    self.test_results = {}

  def forward(self, x):
    self.model.eval()
    return self.model(x)

  def training_step(self, batch, batch_idx):
    images, targets = batch

    # fasterrcnn takes both images and targets for training
    loss_dict = self.model(images, targets)
    loss = self.calculate_model_loss(loss_dict)
    log(f'Training loss {loss} {loss_dict}')
    self.log_dict({'loss': loss}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    images, targets = batch
    log('====validation_step====')
    loss, loss_dict = self.get_model_loss(images, targets)
    log(f'Validation loss_dict {loss} {loss_dict}')
    self.validation_step_loss.append(loss)

    outs = self.model(images)
    # print(f'Targets:\n{targets}')
    # log(f'Outs:\n{outs}')
    predicted_boxes, target_boxes = self.get_step_boxes(outs, targets)
    iou, giou = self.calculate_iou_giou(predicted_boxes, target_boxes)
    self.validation_step_iou.append(iou)
    self.validation_step_giou.append(giou)
    return loss

  def on_validation_epoch_end(self):
    log("====On validation epoch end ====")
    avg_loss = torch.stack(self.validation_step_loss).mean()
    avg_iou = torch.stack(self.validation_step_iou).mean()
    avg_giou = torch.stack(self.validation_step_giou).mean()
    
    self.log_dict({
      'loss': avg_loss,
      'validation_iou': avg_iou.item(),
      'validation_giou': avg_giou.item(),
      }, on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)

    # Evaluation step after all epochs of training have completed
  def test_step(self, test_batch, batch_idx):
    images, targets = test_batch
    loss, loss_dict = self.get_model_loss(images, targets)
    self.test_step_loss.append(loss)

    log(f'Loss:\n{loss}')

    outs = self.model(images)
    # print(f'Targets:\n{targets}')
    # log(f'Outs:\n{outs}')
    predicted_boxes, target_boxes = self.get_step_boxes(outs, targets)
    iou, giou = self.calculate_iou_giou(predicted_boxes, target_boxes)
    self.test_step_iou.append(iou)
    self.test_step_giou.append(giou)
    self.test_step_predicted_boxes.extend([t.tolist() for t in predicted_boxes])
    self.test_step_target_boxes.extend([t.tolist() for t in target_boxes])
    for t in targets:
      self.test_step_image_paths.append(t['img_path'])
    self.test_step_predicted_scores = ColorBarSegmentationSystem.get_top_scores(outs)
    print(f'Test step iou {iou}, giou {giou}')
    return giou

  def on_test_epoch_end(self):
    avg_iou = torch.stack(self.test_step_iou).mean()
    avg_giou = torch.stack(self.test_step_giou).mean()
    avg_loss = torch.stack(self.test_step_loss).mean()
    results = {
      'loss': avg_loss.item(),
      'test_iou': avg_iou.item(),
      'test_giou': avg_giou.item(),
      }

    self.log_dict(results, on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)
    self.test_results = results

  # Get loss from the model, and then weight the different types of loss according to config
  def get_model_loss(self, images, targets):
    # Enabling training mode so we get the same model specific metrics as during training for comparison
    self.model.train()
    # Disable gradiants so that the model does not learn from the validation data
    with torch.no_grad():
      loss_dict = self.model(images, targets)
    log(f'Model loss_dict {loss_dict}')
    loss = self.calculate_model_loss(loss_dict)
    self.model.eval()
    return loss, loss_dict

  def calculate_model_loss(self, loss_dict):
    return sum(self.config.loss_weights[key] * loss for key, loss in loss_dict.items())

  # calculate the intersection of union metric for the best predicted boxes
  def calculate_iou_giou(self, target_boxes, predicted_boxes):
    # log(f'Target boxes:\n{target_boxes}')
    # log(f'Predicted:\n{predicted_boxes}')

    try:
      iou = torch.stack([evaluate_iou(t, o) for t, o in zip(target_boxes, predicted_boxes)]).mean()
      giou = torch.stack([evaluate_giou(t, o) for t, o in zip(target_boxes, predicted_boxes)]).mean()
      return (iou, giou)
    except Exception as e:
      log(f'Failed to calculate IOU/GIOU for targets:\n{target_boxes}\nPrediced:\n{predicted_boxes}\nError was:\n{e}')
    return (0, 0)

  # Takes output from the model for one item, and selects the bounding box with
  # the highest score, assuming its higher than the minimum score threshold
  def get_top_predicted(config, out_entry):
    threshold = config.predict_rounding_threshold
    scores = out_entry['scores']
    if not torch.any(scores > threshold):
      return {
      'boxes' : torch.zeros((0, 4), dtype=torch.float32),
      'labels' : torch.tensor([]),
      'scores' : torch.zeros((0, 4), dtype=torch.float32)
    }

    top_index = scores.argmax().item()
    return {
      'boxes' : out_entry['boxes'][top_index].unsqueeze(0),
      'labels' : torch.tensor([1]),
      'scores' : out_entry['scores'][top_index].unsqueeze(0)
    }

  def get_top_scores(outs):
    top = []
    for entry in outs:
      scores = entry['scores']
      if scores.shape[0] == 0:
        top.append(0)
      else:
        top.append(scores.max().item())
    return top

  # extract predicted and target bounding boxes
  def get_step_boxes(self, outs, targets):
    target_boxes = [next(iter(t['boxes']), torch.zeros((0, 4), dtype=torch.float32, device=self.device)) for t in targets]
    top_predicted = [ColorBarSegmentationSystem.get_top_predicted(self.config, o) for o in outs]
    predicted_boxes = [next(iter(o['boxes']), torch.zeros((0, 4), dtype=torch.float32, device=self.device)) for o in top_predicted]
    return target_boxes, predicted_boxes

  # return prediction results from the test run
  def get_test_set_predictions(self):
    return {
      'img_paths' : self.test_step_image_paths,
      'predicted_boxes' : self.test_step_predicted_boxes,
      'predicted_scores' : self.test_step_predicted_scores,
      'target_boxes' : self.test_step_target_boxes,
    }

  def configure_optimizers(self):
    return torch.optim.SGD(self.model.parameters(), lr=self.config.lr,
                           momentum=0.9, weight_decay=self.config.weight_decay,)

  def record_val_incorrect_predictions(self, dataset):
    return pd.DataFrame(data={})

  def record_test_incorrect_predictions(self, dataset):
    return pd.DataFrame(data={})
