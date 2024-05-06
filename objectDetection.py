import io
import matplotlib.pyplot as plt
import requests
# import inflect
from PIL import Image
from transformers import pipeline


def load_image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)


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

def detect_objects():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    # resized_img = dinner_image.resize((569, 491))
    od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")
    detection_result = od_pipe(dinner_image)
    # print(detection_result)
    processed_image = render_results_in_image(dinner_image, detection_result)
    processed_image.show()


detect_objects()
