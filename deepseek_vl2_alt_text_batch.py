import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

from src.utils.image_normalizer import ImageNormalizer
from src.utils.classifier_config import ClassifierConfig

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from timeit import default_timer as timer
import csv
import os
from pathlib import Path
from PIL import Image

start = timer()
model_name = 'deepseek-vl2-tiny'
model_path = "deepseek-ai/deepseek-vl2-tiny"
# processor = AutoProcessor.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda:0" if torch.cuda.is_available() else "cpu"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)
processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path)
model = vl_gpt.to(torch.bfloat16).cuda().eval()

end = timer()
print(f'Loaded model in {end - start}')

image_dir = "/proj/casden_lab/alt_text/prelim_examples"
output_dir = f'/work/groups/unc_libraries_dcr/boxctron/{model_name}/'
os.makedirs(output_dir, exist_ok = True)

# config = ClassifierConfig()
# config.max_dimension = 800
# config.min_dimension = 800
# config.output_base_path = Path(output_dir)
# config.src_base_path = Path(image_dir)
# normalizer = ImageNormalizer(config)

def caption_image(image_path, system_prompt=None, context=None, max_new_tokens=128):
    global model
    global normalizer
    global processor

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n<|ref|>{context}<|/ref|>.",
            "images": [image_path]
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # norm_path = normalizer.process(image_path)
    # raw_image = Image.open(norm_path).convert("RGB")
    pil_images = load_pil_images(conversation)

    inputs = processor(
        conversations=conversation,
        images=pil_images,
        # force_batchify=True,
        system_prompt=system_prompt
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    # run the model to get the response
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"*****Caption:\n{inputs['sft_format'][0]}", answer)

    return answer

def summarize_description(system_prompt=None, prompt=None, max_new_tokens=128):
    global model
    global normalizer
    global processor

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<|ref|>{prompt}<|/ref|>.",
            "images": [],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    inputs = processor(
        conversations=conversation,
        images=[],
        system_prompt=system_prompt
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    # run the model to get the response
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"*****Summary:\n {inputs['sft_format'][0]}", answer)

    return answer

def generate_system_prompt0():
    return f"""You are a helpful alt text generator assisting visually impaired users."""

def generate_system_prompt1():
    return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. You do no start your response with phrases like 'the image' or 'the picture' unless needed in the context of an illustration or painting type graphic."""

def generate_system_prompt2():
    return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. You do not repeat the prompt you are given, and do not describe what you can or cannot do, you simply generate the alt text. You do no start your response with phrases like generic 'the image' or 'the picture' unless needed in the context of an illustration or painting type graphic."""

def generate_system_prompt3():
    return f"""You are a helpful descriptive metadata generator. You generate clear and detailed descriptions focusing on abstracts that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. You do no start your response with phrases like generic 'the image' or 'the picture' unless needed in the context of an illustration or painting type graphic."""

# def generate_prompt0(row):
#     global collection
#     title = row['Object']
#     format_val = row['Select Format']
#     location = row['Subject Geographic']
#     transcription = row['Transcribed Text on Image']

#     return f"""What is in the image? Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'"""

# def generate_prompt1(row):
#     title = row['Object']
#     format_val = row['Select Format']
#     location = row['Subject Geographic']
#     transcription = row['Transcribed Text on Image']

#     return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Generate alt text for this image which is of type {format_val} and may be related to geographic location {location}. Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'. For context, it has the following title: {title}"""


# def generate_prompt2(row):
#     title = row['Object']
#     format_val = row['Select Format']
#     location = row['Subject Geographic']
#     transcription = row['Transcribed Text on Image']

#     return f"""You are a helpful alt text generator assisting visually impaired users. You generate clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Generate alt text for this image of type {format_val}. Start directly with the key content, avoiding phrases like 'The image shows' or 'This is'"""


# def generate_prompt3(row):
#     title = row['Object']
#     format_val = row['Select Format']
#     location = row['Subject Geographic']
#     transcription = row['Transcribed Text on Image']

#     return f"""Generate alt text for the provided image of type {format_val}. The alt text should be clear and concise captions (10-40 words) that highlight the most important subject and action. Focus only on essential details, avoiding unnecessary background elements. Use simple, everyday language and avoid overly descriptive or poetic words. Start directly with the key content, avoiding leading phrases like 'The image shows', 'This image depicts', 'This is', or similar redundant phrases. Consider that this will be the alt text for the image, so the screen reader already knows it is an image. It can be helpful to describe what type of object is depicted in the image, like if it were a photograph."""

def generate_prompt_full_desc(row):
    title = row['Object']
    format_val = row['Select Format']
    location = row['Subject Geographic']

    return f"""REFERENCE INFORMATION (not in the image):
Geographic location: {location}
Title: {title}
Type: {format_val}

Please provide a detailed and comprehensive description of what you can see in this image, covering:
1. The main subject and its characteristics, focus only on what is clearly visible in the image
2. Any visible text or writing. If text is visible but not fully legible, describe it as "partially legible text" rather than guessing. For handwritten content, only transcribe text that is clearly legible.
3. The type of document or artifact this appears to be
4. Historical context if apparent. Do NOT make assumptions about dates, names, or other specific details unless they are explicitly visible or provided
5. Physical characteristics (condition, color, medium)
6. Any notable details that would help understand the significance of this item
7. Do NOT incorporate the title or geographic information I provided into your description unless these exact words are visibly present in the image
8. Distinguish between what you can actually see and what you might infer


Your complete description will be processed further, so include all relevant visual information without worrying about length."""

def generate_prompt_sum0(full_desc):
    return f"""Summarize the detailed description of the image. Include only the main subject and key details, abd keep it concise: {full_desc}"""

def generate_prompt_sum1(full_desc):
    return f"""Summarize the detailed description of the image in a single sentence. Include the main subject and key details, but keep it concise: {full_desc}"""

def generate_prompt_sum2(full_desc):
    return f"""Summarize the detailed description of the image in a single sentence. Use natural, flowing language while maintaining accuracy about the main subject and key details: {full_desc}"""

def generate_prompt_sum3(full_desc):
    return f"""Summarize this description as alt text for an image, which should be concise but include the main subject and key details: {full_desc}"""

# def generate_prompt2(row):
#     title = row['Object']
#     format_val = row['Select Format']
#     location = row['Subject Geographic']
#     transcription = row['Transcribed Text on Image']

#     return f"""Using the provided image, create an alt-text. The image description should be objective, concise, and descriptive. Descriptions should be straight forward and factual, avoiding interpretations. Begin with a general overview of what the image portrays before providing details. Not everything needs to be described if it is not contextually important. Descriptions should utilize vivid terminology to describe various features like composition, shapes, size, texture, and color. The image alt-text should be concise. A short phrase or at most a couple of sentences. Should the image or graphic contain text, add a section and transcribe all the text presented."""

count = 0

# alt_text_sample_short.csv for only 5 images
with open(Path(image_dir, 'alt_text_sample.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    system_prompt0 = generate_system_prompt0()
    system_prompt1 = generate_system_prompt1()
    system_prompt2 = generate_system_prompt2()
    system_prompt3 = generate_system_prompt3()

    with open(Path(output_dir, f'{model_name}_results.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csvwriter.writerow(['filename', 'text0', 'text1', 'text2', 'text3'])
        csvwriter.writerow(['filename', 'full_desc', 'sum1', 'sum2', 'sum3', 'sum4'])
        ex_row = {'Image Name': 'example', 'Object': 'title', 'Select Format': 'format', 'Subject Geographic': 'location', 'Transcribed Text on Image': 'transcription'}
        # csvwriter.writerow(['prompts', system_prompt0 + generate_prompt0(ex_row),  generate_prompt1(ex_row), generate_prompt2(ex_row), generate_prompt3(ex_row)])
        csvwriter.writerow(['prompts', generate_prompt_full_desc(ex_row), generate_prompt_sum0("full_desc"), generate_prompt_sum1("full_desc"), generate_prompt_sum2("full_desc"), generate_prompt_sum3("full_desc")])

        for row in reader:
            print(f'Processing row {row}')
            image_name = row['Image Name']
            image_path = Path(image_dir, image_name)
            title = row['Object']
            format_val = row['Select Format']
            location = row['Subject Geographic']
            transcription = row['Transcribed Text on Image']
        
            start = timer()
            # prompt0 = generate_prompt0(row)
            # prompt1 = generate_prompt1(row)
            # prompt2 = generate_prompt2(row)
            # prompt3 = generate_prompt3(row)
            prompt0 = generate_prompt_full_desc(row)
            full_desc = caption_image(image_path, system_prompt="", context=prompt0, max_new_tokens=512)
            prompt1 = generate_prompt_sum0(full_desc)
            prompt2 = generate_prompt_sum1(full_desc)
            prompt3 = generate_prompt_sum2(full_desc)
            prompt4 = generate_prompt_sum3(full_desc)
            caption1 = summarize_description(system_prompt="", prompt=prompt1)
            caption2 = summarize_description(system_prompt="", prompt=prompt2)
            caption3 = summarize_description(system_prompt="", prompt=prompt3)
            caption4 = summarize_description(system_prompt="", prompt=prompt4)
            csvwriter.writerow([image_path, full_desc, caption1, caption2, caption3, caption4])
            # caption0 = caption_image(image_path, system_prompt=system_prompt0, context=prompt0)
            # caption1 = caption_image(image_path, system_prompt=system_prompt0, context=prompt1)
            # caption2 = caption_image(image_path, system_prompt=system_prompt0, context=prompt2)
            # caption3 = caption_image(image_path, system_prompt=system_prompt0, context=prompt3)
            # csvwriter.writerow([image_path, caption0, caption1, caption2, caption3])
            end = timer()
            print(f'{count}. Wrote captions to file for {image_path} in {end - start}', flush=True)
            print()
            csvfile.flush()

            count += 1
    