from transformers import BlipForImageTextRetrieval, AutoProcessor
from PIL import Image
import torch

def image_to_text(source_image: Image, text: str):
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    inputs = processor(images=source_image,text=text,return_tensors="pt")
    outputs = model(**inputs)[0]
    print(outputs)
    one_output = torch.nn.functional.softmax(outputs,dim=1)
    print(one_output)
    print(f"Probability of text matching the image: {one_output[0][1]:.4f}")
    print(f"Max Probability: {one_output.max():.4f}")


def app_run():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    image_description = "A couple making dinner"
    image_to_text(dinner_image, image_description)

app_run()