import pytorch_lightning as pl
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import BinaryConfusionMatrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import io
from PIL import Image

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
    self.validation_step_outputs = []
    self.validation_step_labels = []
    self.test_step_loss = []
    self.test_step_acc = []
    self.test_step_outputs = []
    self.test_step_labels = []

    # load model
    fdn_num_filters, self.foundation_model = self.get_foundation_model()
    self.model = self.get_model(fdn_num_filters)

    # We will overwrite this once we run `test()`
    self.test_results = {}

  # Build foundation model for transfer learning, starting from resnet and removing final layer so we can build on top of it
  def get_foundation_model(self):
    foundation = models.resnet50(pretrained=True)
    num_filters = foundation.fc.in_features
    layers = list(foundation.children())[:-1]
    return num_filters, nn.Sequential(*layers)

  # Model maps from the final dimension in resnet (2048 dimensions) down to a single dimension,
  # which is the class of having a color bar or not
  def get_model(self, starting_size):
    model = nn.Sequential(
      nn.Linear(starting_size, 1)
    )
    return model

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
    return optimizer

  # Common step used to process a batch of data (images and labels) through the model,
  # and calculate the loss/accuracy of the model within that batch.
  def _common_step(self, batch, _):
    images, labels = batch
    
    # forward pass using the model
    self.foundation_model.eval()
    # fdn_output = self.foundation_model(images)
    with torch.no_grad():
      fdn_output = self.foundation_model(images).flatten(1)
    logits = self.model(fdn_output)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())

    outputs = None
    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      outputs = preds.squeeze(1)
      num_correct = torch.sum(outputs == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy, outputs, labels

  def training_step(self, train_batch, batch_idx):
    loss, acc, _outputs, _labels = self._common_step(train_batch, batch_idx)
    self.log_dict({'train_loss': loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    loss, acc, outputs, labels = self._common_step(val_batch, batch_idx)
    self.validation_step_loss.append(loss)
    self.validation_step_acc.append(acc)
    self.validation_step_outputs.append(outputs)
    self.validation_step_labels.append(labels)
    return loss, acc

  # Calculate the average loss and accuracy for batches within the validation dataset at the end of a training epoch,
  # and log the results
  def on_validation_epoch_end(self):
    loss_values = torch.stack(self.validation_step_loss)
    avg_loss = torch.stack(self.validation_step_loss).mean()
    avg_acc = torch.stack(self.validation_step_acc).mean()
    
    confusion_data = self.produce_confusion_matrix(self.validation_step_outputs, self.validation_step_labels)
    num_samples = confusion_data.flatten().sum()
    c_percentages = confusion_data / num_samples
    self.log_dict({
      'val_loss': avg_loss,
      'val_acc': avg_acc,
      'val_true_pos': c_percentages[0, 0],
      'val_false_pos': c_percentages[0, 1],
      'val_true_neg': c_percentages[1, 0],
      'val_false_neg': c_percentages[1, 1],
      }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.plot_confusion_matrix('val', confusion_data)
    self.validation_step_loss.clear()
    self.validation_step_acc.clear()
    self.validation_step_outputs.clear()
    self.validation_step_labels.clear()

  # Evaluation step after all epochs of training have completed
  def test_step(self, test_batch, batch_idx):
    loss, acc, outputs, labels = self._common_step(test_batch, batch_idx)
    self.test_step_loss.append(loss)
    self.test_step_acc.append(acc)
    self.test_step_outputs.append(outputs)
    self.test_step_labels.append(labels)
    return loss, acc

  # Produce final report from evaluating the model against the test dataset after training has completed
  def on_test_epoch_end(self):
    avg_loss = torch.stack(self.test_step_loss).mean()
    avg_acc = torch.stack(self.test_step_acc).mean()

    confusion_data = self.produce_confusion_matrix(self.test_step_outputs, self.test_step_labels)
    num_samples = confusion_data.flatten().sum()
    c_percentages = confusion_data / num_samples
    results = {
      'test_loss': avg_loss.item(),
      'test_acc': avg_acc.item(),
      'test_true_pos': c_percentages[0, 0],
      'test_false_pos': c_percentages[0, 1],
      'test_true_neg': c_percentages[1, 0],
      'test_false_neg': c_percentages[1, 1],
      }
    self.log_dict(results,
      on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.plot_confusion_matrix('test', confusion_data)
    self.test_results = results
    self.test_step_loss.clear()
    self.test_step_acc.clear()
    self.test_step_outputs.clear()
    self.test_step_labels.clear()

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
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb = self.logger.experiment
    tb.add_image(phase + "_confusion_matrix", im, global_step=self.current_epoch)
    plt.close()
