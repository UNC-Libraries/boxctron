from PIL import Image
from torchvision import transforms

# normalize color values based on requirements of resnet and convert to a tensor
def load_for_resnet(path):
  input_image = Image.open(path)
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return preprocess(input_image)