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
    self.test_step_iou = []
    self.test_step_giou = []
    # self.validation_step_acc = []
    # self.validation_step_raw_predictions = []
    # self.validation_step_predicted_segments = []
    # self.validation_step_labels = []
    # self.test_step_loss = []
    # self.test_step_acc = []
    # self.test_step_raw_predictions = []
    # self.test_step_predicted_segments = []
    # self.test_step_labels = []

    self.model = fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=3)
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 1
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # load model
    # fdn_num_filters, self.foundation_model = resnet_foundation_model(self.device, getattr(self.config, 'resnet_depth', 50))
    # self.model = self.get_model(fdn_num_filters)

    # We will overwrite this once we run `test()`
    # self.test_results = {}

  def forward(self, x):
    self.model.eval()
    return self.model(x)

  def training_step(self, batch, batch_idx):
    images, targets = batch
    print('=================')
    print(f'Targets {targets}')
    print('=================')
    # for t in targets:
      # print(f'Target {t} {targets[t]}')
    # targets = [{k: v for k, v in t.items()} for t in targets['boxes']]

    # fasterrcnn takes both images and targets for training, returns
    loss_dict = self.model(images, targets)
    loss = sum(loss for loss in loss_dict.values())
    self.log_dict({'train_loss': loss}, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    images, targets = batch
    # fasterrcnn takes only images for eval() mode
    outs = self.model(images)

    iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
    self.validation_step_iou.append(iou)
    giou = torch.stack([evaluate_giou(t, o) for t, o in zip(targets, outs)]).mean()
    self.validation_step_giou.append(giou)
    return giou

  def on_validation_epoch_end(self):
    avg_iou = torch.stack(self.validation_step_iou).mean()
    avg_giou = torch.stack(self.validation_step_giou).mean()
    
    self.log_dict({
      'val_iou': avg_iou,
      'val_giou': avg_giou,
      }, on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)

    # Evaluation step after all epochs of training have completed
  def test_step(self, test_batch, batch_idx):
    images, targets = batch
    outs = self.model(images)
    iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
    self.test_step_iou.append(iou)
    giou = torch.stack([evaluate_giou(t, o) for t, o in zip(targets, outs)]).mean()
    self.test_step_giou.append(giou)
    return giou

  def on_test_epoch_end(self):
    avg_iou = torch.stack(self.test_step_iou).mean()
    avg_giou = torch.stack(self.test_step_giou).mean()
    results = {
      'test_iou': avg_iou,
      'test_giou': avg_giou,
      }

    self.log_dict(results, on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)
    self.test_results = results

  def configure_optimizers(self):
    return torch.optim.SGD(self.model.parameters(), lr=self.config.lr,
                           momentum=0.9, weight_decay=self.config.weight_decay,)
