from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# Load your image
image_path = "/proj/casden_lab/alt_text/prelim_examples/alt13.jpg"
image = Image.open(image_path)
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Provide context along with the image
# context = "Generate This image is titled: Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1"

# Process inputs
# inputs = processor(raw_image, text=context, return_tensors="pt").to("cuda")
inputs = processor(raw_image, return_tensors="pt").to("cuda")

# Generate caption
outputs = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(f'Caption for {image_path}:')
print(caption.strip())