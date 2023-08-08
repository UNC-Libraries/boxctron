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

# System for training a model to classify images as either containing a color bar or not.
# It uses a resnet model as its foundation for transfer learning, then trains on top of that
# for our specific task.
class ColorBarClassifyingSystem(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.validation_step_loss = []
    self.validation_step_acc = []
    self.validation_step_raw_predictions = []
    self.validation_step_predicted_classes = []
    self.validation_step_labels = []
    self.test_step_loss = []
    self.test_step_acc = []
    self.test_step_raw_predictions = []
    self.test_step_predicted_classes = []
    self.test_step_labels = []

    # load model
    fdn_num_filters, self.foundation_model = resnet_foundation_model(self.device, getattr(self.config, 'resnet_depth', 50))
    self.model = self.get_model(fdn_num_filters)

    # We will overwrite this once we run `test()`
    self.test_results = {}

  # Model maps from the final dimension in resnet (2048 dimensions) down to a single dimension,
  # which is the class of having a color bar or not
  def get_model(self, starting_size):
    model = nn.Sequential(
      nn.Linear(starting_size, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, self.config.model_width),
      nn.ReLU(inplace=True),
      nn.Linear(self.config.model_width, 1),
    ).to(self.device)
    return model

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
    return optimizer

  # Forward pass of the provided images through the model
  def forward(self, images):
    # forward pass using the model
    self.foundation_model.eval()
    # fdn_output = self.foundation_model(images)
    with torch.no_grad():
      fdn_output = self.foundation_model(images).flatten(1)
    return self.model(fdn_output)

  # Common step used to process a batch of data (images and labels) through the model,
  # and calculate the loss/accuracy of the model within that batch.
  def _common_step(self, batch, _):
    images, labels = batch
    logits = self.forward(images)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())

    predicted_classes = None
    raw_predictions = None
    with torch.no_grad():
      # Compute accuracy using the logits and labels
      raw_predictions = torch.sigmoid(logits)
      predicted_classes = torch.where(raw_predictions > self.config.predict_rounding_threshold, 1, 0)
      predicted_classes = predicted_classes.squeeze(1)
      num_correct = torch.sum(predicted_classes == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy, raw_predictions, predicted_classes, labels

  def training_step(self, train_batch, batch_idx):
    loss, acc, _raw_predictions, predicted_classes, labels = self._common_step(train_batch, batch_idx)
    confusion_vector = predicted_classes / labels
    false_pos_rate = (torch.sum(confusion_vector == float('inf')).item()) / labels.size(dim=0)
    train_fp_loss = self.loss_with_false_positives(loss, false_pos_rate)
    self.log_dict({'train_loss': loss, 'train_fp_loss': train_fp_loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    # Clear running stats at the beginning of each epoch
    if batch_idx == 0:
      self.validation_step_loss.clear()
      self.validation_step_acc.clear()
      self.validation_step_raw_predictions.clear()
      self.validation_step_predicted_classes.clear()
      self.validation_step_labels.clear()
    loss, acc, raw_predictions, predicted_classes, labels = self._common_step(val_batch, batch_idx)
    self.validation_step_loss.append(loss)
    self.validation_step_acc.append(acc)
    self.validation_step_raw_predictions.append(raw_predictions)
    self.validation_step_predicted_classes.append(predicted_classes)
    self.validation_step_labels.append(labels)
    return loss

  # Calculate the average loss and accuracy for batches within the validation dataset at the end of a training epoch,
  # and log the results
  def on_validation_epoch_end(self):
    avg_loss = torch.stack(self.validation_step_loss).mean()
    avg_acc = torch.stack(self.validation_step_acc).mean()
    
    confusion_data = self.produce_confusion_matrix(self.validation_step_predicted_classes, self.validation_step_labels)
    num_samples = confusion_data.flatten().sum()
    c_percentages = confusion_data / num_samples
    self.log_dict({
      'val_loss': avg_loss,
      'val_fp_loss': self.loss_with_false_positives(avg_loss, c_percentages[0, 1]),
      'val_acc': avg_acc,
      'val_true_pos': c_percentages[1, 1],
      'val_false_pos': c_percentages[0, 1],
      'val_true_neg': c_percentages[0, 0],
      'val_false_neg': c_percentages[1, 0],
      }, on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)
    self.plot_confusion_matrix('val', confusion_data)
    self.plot_precision_recall_curve('val', self.validation_step_raw_predictions, self.validation_step_labels)

  # Special loss value for validation which punishes outcomes for having higher rates of false positives
  def loss_with_false_positives(self, loss, false_pos_rate, weight = 0.3):
    return (loss + (false_pos_rate * weight)).detach().cpu().numpy().item()

  # Evaluation step after all epochs of training have completed
  def test_step(self, test_batch, batch_idx):
    loss, acc, raw_predictions, predicted_classes, labels = self._common_step(test_batch, batch_idx)
    self.test_step_loss.append(loss)
    self.test_step_acc.append(acc)
    self.test_step_raw_predictions.append(raw_predictions)
    self.test_step_predicted_classes.append(predicted_classes)
    self.test_step_labels.append(labels)
    return loss

  # Produce final report from evaluating the model against the test dataset after training has completed
  def on_test_epoch_end(self):
    avg_loss = torch.stack(self.test_step_loss).mean()
    avg_acc = torch.stack(self.test_step_acc).mean()

    confusion_data = self.produce_confusion_matrix(self.test_step_predicted_classes, self.test_step_labels)
    num_samples = confusion_data.flatten().sum()
    c_percentages = confusion_data / num_samples
    results = {
      'test_loss': avg_loss.item(),
      'test_fp_loss': self.loss_with_false_positives(avg_loss, c_percentages[0, 1]),
      'test_acc': avg_acc.item(),
      'test_true_pos': c_percentages[1, 1],
      'test_false_pos': c_percentages[0, 1],
      'test_true_neg': c_percentages[0, 0],
      'test_false_neg': c_percentages[1, 0],
      }
    self.log_dict(results,
      on_step=False, on_epoch=True, prog_bar=self.config.enable_progress_bar, logger=True)
    self.plot_confusion_matrix('test', confusion_data)
    self.plot_precision_recall_curve('test', self.test_step_raw_predictions, self.test_step_labels)
    self.test_results = results

  def predict_step(self, batch, _):
    logits = self.model(self.foundation_model(batch[0]))
    probs = torch.sigmoid(logits)
    return probs

  def produce_confusion_matrix(self, step_outputs, step_labels):
    outputs = torch.cat([o for o in step_outputs])
    labels = torch.cat([l for l in step_labels])
    bcm = BinaryConfusionMatrix().to(self.device)
    return bcm(outputs, labels).detach().cpu().numpy().astype(int)

  def plot_confusion_matrix(self, phase, confusion_data):
    # confusion matrix
    df_cm = pd.DataFrame(confusion_data, range(2), range(2))
    plt.title('Validation Confusion Matrix', fontsize = 16)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 16}) # font size
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('True Label', fontsize = 15)
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    self.logger.experiment.add_image(phase + '_confusion_matrix', im, global_step=self.current_epoch)
    plt.close()

  def plot_precision_recall_curve(self, phase, step_raw_predictions, step_labels):
    raw_predictions = torch.cat([o for o in step_raw_predictions])
    labels = torch.cat([l for l in step_labels])
    prec, recall, thresholds = precision_recall_curve(labels.cpu().numpy(), raw_predictions.cpu().numpy())
    plt.title(f'P/R Thresholds {phase} {self.current_epoch}')
    plt.plot(thresholds, prec[:-1], 'b', label='Precision')
    plt.plot(thresholds, recall[:-1], 'y', label='Recall')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(which='both', axis='both', color='gray', linestyle='-', linewidth=1)
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    self.logger.experiment.add_image(phase + '_prec_recall', im, global_step=self.current_epoch)
    plt.close()

  def record_val_incorrect_predictions(self, dataset):
    return self.record_incorrect_predictions(dataset, self.validation_step_raw_predictions, self.validation_step_predicted_classes, self.validation_step_labels)

  def record_test_incorrect_predictions(self, dataset):
    return self.record_incorrect_predictions(dataset, self.test_step_raw_predictions, self.test_step_predicted_classes, self.test_step_labels)

  def record_incorrect_predictions(self, dataset, step_raw_preds, step_preds, step_labels):
    raw_preds = torch.cat([o for o in step_raw_preds]).flatten()
    predictions = torch.cat([o for o in step_preds])
    labels = torch.cat([l for l in step_labels])

    errors = (predictions - labels != 0)
    error_indices = torch.where(errors == True)[0].detach().cpu().numpy()
    error_paths = [str(dataset.image_paths[i]) for i in error_indices]
    
    error_true_labels = labels[errors]
    error_pred_labels = predictions[errors]
    error_raw_preds = raw_preds[errors]
    gap_true_vs_pred = torch.abs(error_true_labels.float() - error_raw_preds)
    results = {
      'true_labels': error_true_labels.detach().cpu().numpy(),
      'pred_labels': error_pred_labels.detach().cpu().numpy(),
      'raw_preds': error_raw_preds.detach().cpu().numpy(),
      'gap_true_vs_pred': gap_true_vs_pred.detach().cpu().numpy(),
      'image_paths': error_paths
    }
    results_df = pd.DataFrame(data=results)
    return results_df.sort_values(by=['gap_true_vs_pred'])
