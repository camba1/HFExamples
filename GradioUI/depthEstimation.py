import gradio as gr
from transformers import pipeline
from PIL import Image
import torch
import numpy as np


def depth_estimation(source_image: Image) -> Image:
    pipe = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas")
    output = pipe(source_image)
    print(output)
    print(output["predicted_depth"])
    print(output["predicted_depth"].unsqueeze(1).shape)
    prediction = torch.nn.functional.interpolate(output["predicted_depth"].unsqueeze(1),
                                                 size=source_image.size[::-1],
                                                 mode="bicubic",
                                                 align_corners=False)
    print(prediction.shape)
    print(source_image.size[::-1])
    print(prediction)

    # We need to convert the tensor to an image by normalizing it and converting it to uint8
    prediction_output = prediction.squeeze().numpy()
    formatted = (prediction_output * 255 / np.max(prediction_output)).astype(np.uint8)
    depth = Image.fromarray(formatted)
    return depth


demo = gr.Interface(fn=depth_estimation, inputs=gr.Image(type="pil"), outputs=gr.Image(type="pil"))

demo.launch()
