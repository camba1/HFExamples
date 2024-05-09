import os
import warnings
import gradio as gr
from transformers import pipeline
from transformers.utils import logging

# Ignore warning that are not application related
logging.set_verbosity_error()
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# Load model
image_captioning = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Perform inference
def launch(input):
    out= image_captioning(input)
    return out[0]['generated_text']

# Create the gradio interface
int_face = gr.Interface(
    launch,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

# Run the gradio interface
int_face.launch()