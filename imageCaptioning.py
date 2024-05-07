from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
# import torch

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


def image_captioning(source_image: Image, text: str = None):
    inputs = processor(text=text, images=source_image, return_tensors="pt")
    print(inputs)
    outputs = model.generate(**inputs)
    # The outputs are text tokens that need to be decoded into text
    print(outputs)
    print(processor.decode(outputs[0], skip_special_tokens=True))


def app_run():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    start_of_reply = "A photograph of"   # optional It just ensures the llm answer starts with this sentence
    image_captioning(dinner_image, start_of_reply)
    image_captioning(dinner_image)


app_run()