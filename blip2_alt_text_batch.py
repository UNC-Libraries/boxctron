from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from timeit import default_timer as timer
import csv
from pathlib import Path
from src.utils.image_normalizer import ImageNormalizer
from src.utils.classifier_config import ClassifierConfig
import os

start = timer()
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16
)
    #device_map="auto")
device = next(model.parameters()).device
end = timer()
print(f'Loaded model in {end - start}')
image_dir = "/proj/casden_lab/alt_text/prelim_examples"
output_dir = '/work/groups/unc_libraries_dcr/boxctron/blip2/'
os.makedirs(output_dir, exist_ok = True)

config = ClassifierConfig()
config.max_dimension = 800
config.min_dimension = 800
config.output_base_path = Path(output_dir)
config.src_base_path = Path(image_dir)
normalizer = ImageNormalizer(config)

collection = "North Carolina County Photographic Collection, circa 1850-2000"
series = "Series 1. Photographic Prints, circa 1850-2000. / Subseries 1.11. Buncombe County"

def caption_image(image_path, context=None, max_new_tokens=50):
    global model
    global processor
    global normalizer
    global device
    
    if context is None:
        prompt = None
    else:
        prompt = f'Question: {context} Answer:'
    
    norm_path = normalizer.process(image_path)
    # return f'test {norm_path}'
    raw_image = Image.open(norm_path)
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, 
        max_new_tokens=75,      # Longer to allow more detail
        min_length=25,           # Ensure a minimum length
        do_sample=True,          # Enable sampling for more varied outputs
        temperature=0.1,         # Lower temperature for more factual output
        num_beams=15,             # Beam search for better quality
        no_repeat_ngram_size=3,   # Prevent repetition
        repetition_penalty=1.4
        )
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    if prompt is not None:
        caption = caption.replace(prompt, '').strip()
    return caption

def generate_prompt1(row):
    global collection
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""You are a helpful alt text generator assisting visually impaired users. Generate a clear and concise caption (10-20 words) that highlights the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words."""

    # return f"""Create alt text for this {format} titled "{title}" from geographic location {location} in collection and time period {collection}. Describe in 1-2 sentences what you can see in the image, including notable buildings, people, vehicles, and activities. Only include what is visibly present in the image."""

def generate_prompt2(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""You are a helpful alt text generator assisting visually impaired users. Generate a clear and concise caption (10-20 words) that highlights the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Consider this context before analyzing the image: {format} titled "{title}" from geographic location {location}"""

    # return f"""Create alt text for this {format} titled "{title}" from geographic location {location}. Describe in 1-2 sentences what you can see in the image, including notable objects, people, and activities. Only include what is visibly present in the image."""

def generate_prompt3(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""Using the provided image, create an alt-text. The image description should be objective, concise, and descriptive. Descriptions should be straight forward and factual, avoiding interpretations. Begin with a general overview of what the image portrays before providing details. Not everything needs to be described if it is not contextually important. Descriptions should utilize vivid terminology to describe various features like composition, shapes, size, texture, and color. Avoid using picture of, image of, and photo of unless needed in the context of an illustration or painting type graphic. The image alt-text should be concise. A short phrase or at most a couple of sentences. Should the image or graphic contain text, add a section and transcribe all the text presented. The image is titled "{title}" from geographic location {location}"""
    # return f"""Create alt text that describes the visual content for this {format} titled "{title}" from geographic location {location}. Describe in 1-2 sentences what you can see in the image, including notable objects, people, and activities. Only include what is visibly present in the image or the details in this prompt."""


count = 0

with open(Path(image_dir, 'alt_text_sample.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    with open(Path(output_dir, 'blip2_results.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['filename', 'text1', 'text2', 'text3', 'text4'])
        ex_row = {'Image Name': 'example', 'Object': 'title', 'Select Format': 'format', 'Subject Geographic': 'location', 'Transcribed Text on Image': 'transcription'}
        csvwriter.writerow(['prompts', '', generate_prompt1(ex_row), generate_prompt2(ex_row), generate_prompt3(ex_row)])

        for row in reader:
            print(f'Processing row {row}')
            image_name = row['Image Name']
            image_path = Path(image_dir, image_name)
            title = row['Object']
            format_val = row['Select Format']
            location = row['Subject Geographic']
            transcription = row['Transcribed Text on Image']
        
            start = timer()
            prompt1 = generate_prompt1(row)
            prompt2 = generate_prompt2(row)
            prompt3 = generate_prompt3(row)
            caption0 = caption_image(image_path)
            caption1 = caption_image(image_path, context=prompt1)
            caption2 = caption_image(image_path, context=prompt2)
            caption3 = caption_image(image_path, context=prompt3)
            csvwriter.writerow([image_path, caption0, caption1, caption2, caption3])
            end = timer()
            print(f'{count}. Wrote captions to file for {image_path} in {end - start}')
            print()
            csvfile.flush()

            count += 1
    