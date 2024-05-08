from transformers import CLIPModel, AutoProcessor
from PIL import Image
# import warnings
# warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
# warnings.filterwarnings(action="ignore", category=FutureWarning)


def zero_shot_image_classifier(source_image: Image, labels: list[str]):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # We add padding=True to make sure the labels are padded to the same length
    inputs = processor(text=labels, images=source_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    # print(outputs)
    # the logits per image represent the image-text similarity scores
    print(outputs.logits_per_image)
    # Convert to a list of probabilities
    probabilities = outputs.logits_per_image.softmax(dim=1)[0]
    print(probabilities)
    for i in  range(len(labels)):
        print(f"{labels[i]} with probability {probabilities[i]:.4f}")


def app_run():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    labels = ["photo of a couple making dinner", "a couple at the beach", "Cooking dinner"]
    zero_shot_image_classifier(dinner_image, labels)


app_run()
