from PIL import Image
from torchvision import transforms, from_numpy
import torchvision.models as models
import torch.nn as nn

# normalize color values based on requirements of resnet and convert to a tensor
def load_for_resnet(path, max_dimension):
  input_image = Image.open(path)
  preprocess = transforms.Compose([
      # This will crop OR pad images out to a square of size max_dimension
      transforms.CenterCrop(max_dimension),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return preprocess(input_image)

# Transform mask np.Array based on the requirements of resnet and convert to a tensor
def load_mask_for_resnet(mask_array, max_dimension):
  mask = from_numpy(mask_array).float()
  preprocess = transforms.Compose([
      # Crops masks with identical transformations to images
      transforms.CenterCrop(max_dimension),
  ])
  return preprocess(mask)

def resnet_foundation_model(device, resnet_depth = 50):
  foundation = None
  if resnet_depth == 50:
    foundation = models.resnet50(weights='DEFAULT').to(device)
  elif resnet_depth == 18:
    foundation = models.resnet18(weights='DEFAULT').to(device)
  num_filters = foundation.fc.in_features
  layers = list(foundation.children())[:-1]
  return num_filters, nn.Sequential(*layers)