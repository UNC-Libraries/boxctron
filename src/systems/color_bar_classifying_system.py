import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

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
    self.test_step_loss = []
    self.test_step_acc = []

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

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      num_correct = torch.sum(preds.squeeze(1) == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy

  def training_step(self, train_batch, batch_idx):
    loss, acc = self._common_step(train_batch, batch_idx)
    self.log_dict({'train_loss': loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    loss, acc = self._common_step(val_batch, batch_idx)
    self.validation_step_loss.append(loss)
    self.validation_step_acc.append(acc)
    return loss, acc

  # Calculate the average loss and accuracy for batches within the validation dataset at the end of a training epoch,
  # and log the results
  def on_validation_epoch_end(self):
    avg_loss = torch.stack(self.validation_step_loss).mean()
    avg_acc = torch.stack(self.validation_step_acc).mean()
    self.log_dict({'val_loss': avg_loss, 'val_acc': avg_acc},
      on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.validation_step_loss.clear()
    self.validation_step_acc.clear()

  # Evaluation step after all epochs of training have completed
  def test_step(self, test_batch, batch_idx):
    loss, acc = self._common_step(test_batch, batch_idx)
    self.test_step_loss.append(loss)
    self.test_step_acc.append(acc)
    return loss, acc

  # Produce final report from evaluating the model against the test dataset after training has completed
  def on_test_epoch_end(self):
    avg_loss = torch.mean(torch.stack([o for o in self.test_step_loss]))
    avg_acc = torch.mean(torch.stack([o for o in self.test_step_acc]))

    results = {'test_loss': avg_loss.item(), 'test_acc': avg_acc.item()}
    self.log_dict(results,
      on_step=False, on_epoch=True, prog_bar=True, logger=True)

    self.test_results = results
    self.test_step_loss.clear()
    self.test_step_acc.clear()

  def predict_step(self, batch, _):
    logits = self.model(self.foundation_model(batch[0]))
    probs = torch.sigmoid(logits)
    return probs