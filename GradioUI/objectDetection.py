import io
import gradio as gr
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")


def render_results_in_image(in_pil_img, in_results):
    plt.figure(figsize=(16, 10))
    plt.imshow(in_pil_img)

    ax = plt.gca()

    for prediction in in_results:
        x, y = prediction['box']['xmin'], prediction['box']['ymin']
        w = prediction['box']['xmax'] - prediction['box']['xmin']
        h = prediction['box']['ymax'] - prediction['box']['ymin']

        ax.add_patch(plt.Rectangle((x, y),
                                   w,
                                   h,
                                   fill=False,
                                   color="green",
                                   linewidth=2))
        ax.text(
            x,
            y,
            f"{prediction['label']}: {round(prediction['score'] * 100, 1)}%",
            color='red'
        )

    plt.axis("off")

    # Save the modified image to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',
                bbox_inches='tight',
                pad_inches=0)
    img_buf.seek(0)
    modified_image = Image.open(img_buf)

    # Close the plot to prevent it from being displayed
    plt.close()

    return modified_image


def get_pipeline_prediction(pil_image):
    pipeline_output = od_pipe(pil_image)
    processed_image = render_results_in_image(pil_image, pipeline_output)
    return processed_image


demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image", type="pil"),
  outputs=gr.Image(label="Output image with predicted instances",type="pil")
)

demo.launch()