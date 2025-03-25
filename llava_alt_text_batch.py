import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from timeit import default_timer as timer
from src.utils.image_normalizer import ImageNormalizer
from src.utils.classifier_config import ClassifierConfig
import csv
from pathlib import Path

def load_model(model_id):
    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
                                                              use_flash_attention_2=True,
                                                              torch_dtype=torch.float16) 
    model.to("cuda:0")

    print("Model Generation Config:")
    print(f"Max Length: {model.config.max_length}")
    # print(f"Max Position Embeddings: {model.config.max_position_embeddings}")
    return model, processor

def caption_image(image_path, context=None, max_new_tokens=200):
    global model
    global normalizer
    global processor

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": context},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    norm_path = normalizer.process(image_path)
    raw_image = Image.open(norm_path).convert("RGB")
    inputs = processor(images=raw_image,
                       text=prompt,
                       return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs,
                            do_sample=True,
                            temperature=0.5,
                            max_new_tokens=max_new_tokens)
    answer = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT: " in answer:
        answer = answer.split("ASSISTANT: ", 1)[1]
    print(f"*****Caption:\n {answer}")
    return answer

def summarize_description(prompt=None, max_new_tokens=128):
    global model
    global normalizer
    global processor

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor(text=prompt,
                       return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs,
                            do_sample=True,
                            temperature=0.2,
                            max_new_tokens=max_new_tokens)
    answer = processor.decode(output[0], skip_special_tokens=True)
    if "SUMMARY: " in answer:
        answer = answer.split("SUMMARY: ", 1)[1]
    print(f"*****Summary:\n {answer}")
    return answer

def generate_prompt0(row):
    global collection
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""You are a helpful alt text generator assisting visually impaired users. What is in the image? Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'"""

def generate_prompt1(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Generate alt text for this image which is of type {format_val} and may be related to geographic location {location}. Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'. For context, it has the following title: {title}"""


def generate_prompt2(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']
    transcription = row['Transcribed Text on Image']

    return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Generate alt text for this image of type {format_val}. Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'"""

def generate_prompt_full_desc(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']

    return f"""REFERENCE INFORMATION (not in the image):
Geographic location: {location}
Title: {title}
Type: {format_val}

Please provide a detailed and comprehensive description of what you can see in this image, covering:
The main subject and its characteristics, focus only on what is clearly visible in the image
Transcription of any visible text or writing. If text is visible but not fully legible, describe it as "partially legible text" rather than guessing. For handwritten content, only transcribe text that is clearly legible.
The type of document or artifact this appears to be
Historical context if apparent. Do NOT make assumptions about dates, names, or other specific details unless they are explicitly visible or provided
Physical characteristics (condition, color, medium)
Any notable details that would help understand the significance of this item
Do NOT incorporate the title or geographic information I provided into your description unless these exact words are visibly present in the image
Distinguish between what you can actually see and what you might infer.


Your complete description will be processed further, so include all relevant visual information without worrying about length."""

def generate_prompt_sum0(full_desc):
    return f"""Summarize the detailed description of the image, highlighting the most important subjects and actions. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. If you or the detailed description are unsure about a detail, do not include it in the summary. The description should be concise and include only the main subject and key details:\n\n{full_desc}\n\nSUMMARY:"""

def generate_prompt_sum2(full_desc):
    return f"""Summarize the detailed description of the image in one sentence, highlighting the most important subjects, actions and details. If you or the detailed description are unsure about a something, or it is stated that something is missing or unknown, do not include it in the summary. Use natural, flowing language while maintaining accuracy about the main subject and key details:\n\n{full_desc}\n\nSUMMARY:"""


start = timer()
model_name = 'llava-v1.6-vicuna-13b'
model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
model, processor = load_model(model_id)

end = timer()
print(f'Loaded model in {end - start}')

image_dir = "/proj/casden_lab/alt_text/prelim_examples"
output_dir = f'/work/groups/unc_libraries_dcr/boxctron/{model_name}/'
os.makedirs(output_dir, exist_ok = True)

config = ClassifierConfig()
config.max_dimension = 1000
config.min_dimension = 800
config.output_base_path = Path(output_dir)
config.src_base_path = Path(image_dir)
normalizer = ImageNormalizer(config)

count = 0

input_file = Path(image_dir, 'alt_text_sample.csv')
# input_file = Path(image_dir, 'alt_text_sample_short.csv')
# input_file = Path(output_dir, f'{model_name}_results.csv')
output_file = f'{model_name}_results.csv'
# output_file = f'{model_name}_summary.csv'
# alt_text_sample_short.csv for only 5 images
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    with open(Path(output_dir, output_file), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csvwriter.writerow(['filename', 'text0', 'text1', 'text2', 'text3', 'full_desc', 'sum1', 'sum3'])
        csvwriter.writerow(['filename', 'full_desc', 'sum1', 'sum3'])
        # csvwriter.writerow(['filename', 'full_desc', 'sum1', 'sum2', 'sum3', 'sum4'])
        ex_row = {'Image Name': 'example', 'Object': 'title', 'Select Format': 'format', 'Subject Geographic': 'location', 'Transcribed Text on Image': 'transcription'}
        # csvwriter.writerow(['prompts', generate_prompt0(ex_row),  generate_prompt1(ex_row), generate_prompt2(ex_row), generate_prompt_full_desc(ex_row), generate_prompt_sum0("full_desc"), generate_prompt_sum2("full_desc")])
        csvwriter.writerow(['prompts', generate_prompt_full_desc(ex_row), generate_prompt_sum0("full_desc"), generate_prompt_sum2("full_desc")])
        # csvwriter.writerow(['prompts', generate_prompt_full_desc(ex_row), generate_prompt_sum0("full_desc"), generate_prompt_sum1("full_desc"), generate_prompt_sum2("full_desc"), generate_prompt_sum3("full_desc")])

        for row in reader:
            print(f'Processing row {row}')
            image_name = row['Image Name']
            image_path = Path(image_dir, image_name)
            title = row['Object']
            format_val = row['Select Format']
            location = row['Subject Geographic']
            transcription = row['Transcribed Text on Image']
            # image_path = row['filename']
            # full_desc = row['full_desc']
        
            start = timer()
            # prompt0 = generate_prompt0(row)
            # prompt1 = generate_prompt1(row)
            # prompt2 = generate_prompt2(row)
            # prompt3 = generate_prompt3(row)

            # caption0 = caption_image(image_path, context=prompt0)
            # caption1 = caption_image(image_path, context=prompt1)
            # caption2 = caption_image(image_path, context=prompt2)
            # caption3 = caption_image(image_path, system_prompt=system_prompt0, context=prompt3)
            desc_prompt0 = generate_prompt_full_desc(row)
            full_desc = caption_image(image_path, context=desc_prompt0, max_new_tokens=1024)
            sum_prompt1 = generate_prompt_sum0(full_desc)
            # sum_prompt2 = generate_prompt_sum1(full_desc)
            sum_prompt3 = generate_prompt_sum2(full_desc)
            # sum_prompt4 = generate_prompt_sum3(full_desc)
            sum_caption1 = summarize_description(prompt=sum_prompt1)
            # sum_caption2 = summarize_description(prompt=sum_prompt2)
            sum_caption3 = summarize_description(prompt=sum_prompt3)
            # sum_caption4 = summarize_description(prompt=sum_prompt4)
            # csvwriter.writerow([image_path, full_desc, caption1, caption2, caption3, caption4])
            # csvwriter.writerow([image_path, caption0, caption1, caption2, full_desc, sum_caption1, sum_caption3])
            csvwriter.writerow([image_path, full_desc, sum_caption1, sum_caption3])
            end = timer()
            print(f'{count}. Wrote captions to file for {image_path} in {end - start}', flush=True)
            print()
            csvfile.flush()

            count += 1
    