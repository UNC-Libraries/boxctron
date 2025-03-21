from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from timeit import default_timer as timer


start = timer()
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
end = timer()
print(f'Loaded model in {end - start}')

count = 0

def caption_image(image_path, context=None, max_new_tokens=50):
    global count
    global model
    global processor
    start = timer()
    if context is None:
        prompt = None
    else:
        prompt = f'Question: {context} Answer:'
    raw_image = Image.open(image_path)
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    outputs = model.generate(**inputs, 
        max_new_tokens=150,      # Longer to allow more detail
        min_length=50,           # Ensure a minimum length
        do_sample=True,          # Enable sampling for more varied outputs
        temperature=0.8,         # Lower temperature for more factual output
        num_beams=5,             # Beam search for better quality
        no_repeat_ngram_size=3   # Prevent repetition
        )
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    if prompt is not None:
        caption = caption.replace(prompt, '').strip()
    end = timer()
    print(f'*** {count} Caption for {image_path} generated in {end - start}:')
    print(f'*** With params context={context}, max_new_tokens={max_new_tokens}')
    print('*******************************************')
    print(caption)
    print()
    count += 1
    return caption

# image_path = "/proj/casden_lab/alt_text/prelim_examples/alt13.jpg"
# title = "Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1"
# collection = "North Carolina County Photographic Collection, circa 1850-1990"
# series = "Series 1. Photographic Prints, circa 1850-1990. / Subseries 1.2. Alexander County"
# format = "B&W Photographic Print"
# location = "Alexander County (N.C.) and Taylorsville (N.C.)"

image_path = "/proj/casden_lab/alt_text/prelim_examples/alt10.jpg"
title = "Folder 0125: Asheville: Streets: Scan 01"
collection = "North Carolina County Photographic Collection, circa 1850-2000"
series = "Series 1. Photographic Prints, circa 1850-2000. / Subseries 1.11. Buncombe County"
format = "B&W Illustration"
location = "Buncombe County (N.C.) and Asheville (N.C.)"
transcription = "LOOKING NORTH FROM COURT PLACE."

# caption_image(image_path, context=None)
# caption_image(image_path, model=model, processor=processor, max_new_tokens=100)
# caption_image(image_path, model=model, processor=processor, max_new_tokens=200)
# context = f"What is the alt text for this image which has title '{title}'?"
# caption_image(image_path, context=context)
# caption_image(image_path, context=context, max_new_tokens=100)
# context = f"What is the alt text for this image which has title {title}?"
# caption_image(image_path, context=context)
# context = "What is the alt text for this image which is part to the {collection}?"
# caption_image(image_path, context=context)

# context = f"What is the alt text for this image which has title '{title}', which is part of the {collection}?"
# caption_image(image_path, context=context)
# caption_image(image_path, context=context, max_new_tokens=100)

# context = f"What is the caption for this image which has title '{title}'?"
# caption_image(image_path, context=context)

# context = "What is the alt text for this image?"
# caption_image(image_path, context=context)

# context = "What is the description for this image?"
# caption_image(image_path, context=context)

# context = f"What is the alt text for this {format}?"
# caption_image(image_path, context=context)

# context = f"What is the caption for this {format}?"
# caption_image(image_path, context=context)

# context = f"What is the alt text for this {format} which has title '{title}'?"
# caption_image(image_path, context=context)

# context = f"Using 140 or fewer characters, what is the description for this {format} which has title {title}?"
# caption_image(image_path, context=context)

# context = f"Using 140 or fewer characters, what is the description for this image which has title {title}?"
# caption_image(image_path, context=context)

# context = f"Using 140 or fewer characters, what is the description for this {format} which has title {title} from {location}?"
# caption_image(image_path, context=context)

# context = f"What is the alt text for this {format} which has title {title} from {location}?"
# caption_image(image_path, context=context)

# context = f"What is the alt text for this {format} which has title {title} from {location}?"
# caption_image(image_path, context=context)

# context = f"""Describe what you can see in this image. 

# Important: Only describe what is visibly present in the image. Do not make assumptions about specific dates or information not provided.

# Provide a detailed alt text that describes the visual content, including whether it shows an exterior or interior view, architectural details, and any visible people or objects."""
# caption_image(image_path, context=context)

# context = f"""Describe what you can see in this black and white photographic print titled "{title}". 

# Important: Only describe what is visibly present in the image. Do not make assumptions about specific dates or information not provided. This is a {format} from {location}.

# It is from {series} in collection {collection}.

# Provide a detailed alt text that describes the visual content, including whether it shows an exterior or interior view, architectural details, and any visible people or objects."""
# caption_image(image_path, context=context)


# context = f"""Describe what you can see in this {format} titled "{title}". 

# Important: Only describe what is visibly present in the image or in the information provided here. Do not make assumptions about specific dates, places, people or information not provided. 

# It is related to the location {location}.

# It is from {series} in collection {collection}.

# Provide a detailed alt text that describes the visual content, including whether it shows an exterior or interior view, architectural details, and any visible people or objects."""
# caption_image(image_path, context=context)



# context = f"""Describe what you can see in this {format} titled "{title}". 

# Important: Only describe what is visibly present in the image or in the information provided here. Do not make assumptions about specific dates, places, people or information not provided. 

# It is related to the location {location}.

# It is from {series} in collection {collection}.

# Provide a detailed alt text that describes the visual content, including whether it shows an exterior or interior view, architectural details, type of media displyed, and any visible people or objects."""
# caption_image(image_path, context=context)


# context = f"""Describe what you can see in this image. 

# Important: Only describe what is visibly present in the image and the following details:
# * It is related to the location {location}.
# * It is from {series} in collection {collection}.
# * The format is {format}.
# * The image was given the title "{title}".

# Do not make assumptions about any other specific dates, places, people or information not provided. 

# Provide a detailed alt text that describes the visual content, including whether it shows an exterior or interior view, architectural details, type of media displyed, and any visible people or objects."""
# caption_image(image_path, context=context)


# context = f"""Provide a detailed alt text of this image that describes the visual content, including whether it shows an exterior or interior view, architectural details, type of media displyed, and any visible people or objects.

# Important: Only describe what is visibly present in the image and the following details:
# * It is related to the location {location}.
# * It is from {series} in collection {collection}.
# * The format is {format}.
# * The image was given the title "{title}".

# Do not make assumptions about any other specific dates, places, people or information not provided."""
# caption_image(image_path, context=context)



context = f"""Provide a detailed alt text of this image that describes the visual content, including whether it shows an exterior or interior view, architectural details, type of media displyed, and any visible people or objects.

Important: Only describe what is visibly present in the image and the following details:
* It is related to the location {location}.
* It is from {series} in collection {collection}.
* The format is {format}.
* The image was given the title "{title}"."""
caption_image(image_path, context=context)


# context = f"""Provide a detailed alt text of this image that describes the visual content, including whether it shows an exterior or interior view, architectural details, type of media displyed, and any visible people or objects.

# Important: Only describe what is visibly present in the image and the following details:
# * It is related to the location {location}.
# * It is from {series} in collection {collection}.
# * The format is {format}.
# * The image was given the title "{title}".
# * The image includes the transcription: "{transcription}".

# Do not make assumptions about any other specific dates, places, people or information not provided."""
# caption_image(image_path, context=context)


# context = f"""Generate detailed alt text for this image.

# Available metadata:
# - Title: "{title}"
# - Geographic location: {location}
# - Collection and time period: {collection}
# - Series: {series}
# - Format: {format}
# - Transcription: "{transcription}"

# Describe the main subject and important visual elements visible in the image. Include relevant details about the setting, people, objects, and activities if present. Only describe what you can actually see in the image. Do not make assumptions beyond the provided metadata."""
# caption_image(image_path, context=context)


context = f"""Create alt text for this {format} titled "{title}" from geographic location {location}.

Describe in 1-2 sentences what you can see in the image, including notable buildings, people, vehicles, and activities. Only include what is visibly present in the image."""
caption_image(image_path, context=context)



context = f"""Create alt text for this {format} titled "{title}" from geographic location {location} in collection and time period {collection}.

Describe in 1-2 sentences what you can see in the image, including notable buildings, people, vehicles, and activities. Only include what is visibly present in the image."""
caption_image(image_path, context=context)

# context = "Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1. North Carolina County Photographic Collection, circa 1850-1990."
# caption_image(image_path, context=context)
# context = "Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1. North Carolina County Photographic Collection, circa 1850-1990. Series 1. Photographic Prints, circa 1850-1990. / Subseries 1.2. Alexander County"
# caption_image(image_path, context=context)
# context = "This image is title: Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1. It is from collection: North Carolina County Photographic Collection, circa 1850-1990. Series 1. Photographic Prints, circa 1850-1990. / Subseries 1.2. Alexander County"
# caption_image(image_path, context=context)
# context = "Generate alt text for image with title Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1"
# caption_image(image_path, context=context)
# context = "Generate alt text for the provided image which has the title Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1"
# caption_image(image_path, context=context)
# context = "Generate alt text which is no more than 140 words long, for the provided image which has the title Folder 0006: Taylorsville, Alexander County: Courthouse: Scan 1"
# caption_image(image_path, context=context)