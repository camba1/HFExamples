from transformers import BlipForQuestionAnswering , AutoProcessor
# from transformers.utils import logging
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
warnings.filterwarnings(action="ignore", category=FutureWarning)
# logging.set_verbosity_error()


model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

def visual_question_answering(source_image: Image, text: str):
    inputs = processor(text=text, images=source_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    print(outputs)
    print(processor.decode(outputs[0], skip_special_tokens=True))


def app_run():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    question = "How many people are in this image?"
    visual_question_answering(dinner_image, question)
    question = "what is the color of the t-shirt used by the male in the image?"
    visual_question_answering(dinner_image, question)
app_run()