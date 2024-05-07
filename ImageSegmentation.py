from transformers import pipeline
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# segmentation from a point imports
from transformers import SamModel, SamProcessor


class Utils:
    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0),
                                   w,
                                   h, edgecolor='green',
                                   facecolor=(0, 0, 0, 0),
                                   lw=2))

    def show_boxes_on_image(self, raw_image, boxes):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_on_image(self, raw_image, input_points, input_labels=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_and_boxes_on_image(self, raw_image,
                                       boxes,
                                       input_points,
                                       input_labels=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis('on')
        plt.show()

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0],
                   pos_points[:, 1],
                   color='green',
                   marker='*',
                   s=marker_size,
                   edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0],
                   neg_points[:, 1],
                   color='red',
                   marker='*',
                   s=marker_size,
                   edgecolor='white',
                   linewidth=1.25)

    @staticmethod
    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def show_mask_on_image(self, raw_image, mask, return_image=False):
        if not isinstance(mask, torch.Tensor):
            mask = torch.Tensor(mask)

        if len(mask.shape) == 4:
            mask = mask.squeeze()

        fig, axes = plt.subplots(1, 1, figsize=(15, 15))

        mask = mask.cpu().detach()
        axes.imshow(np.array(raw_image))
        self.show_mask(mask, axes)
        axes.axis("off")
        plt.show()

        if return_image:
            fig = plt.gcf()
            return self.fig2img(fig)

    def show_pipe_masks_on_image(self, raw_image, outputs):
        """
        Draw all the masks on the image
        :param raw_image: Image to draw on
        :param outputs: outputs of the segmentation model to overlay on the image
        """
        plt.imshow(np.array(raw_image))
        ax = plt.gca()
        for mask in outputs["masks"]:
            self.show_mask(mask, ax=ax, random_color=True)
        plt.axis("off")
        plt.show()


img_utils = Utils()


def mask_generation(source_image: Image):
    # We use a segment anything model (SAM) for mask generation
    sam_pipe = pipeline("mask-generation", model="Zigeng/SlimSAM-uniform-77")
    # THe higher the points_per_batch, the more efficient the inference will be,
    # but also the longer it will take
    output = sam_pipe(source_image, points_per_batch=32)

    # Show all masks found by the model on the image
    img_utils.show_pipe_masks_on_image(source_image, output)


def mask_generation_from_single_point(source_image: Image):
    model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
    # This is were we will try to get a mask from
    # We can also use a list of points
    input_points = [[[400, 226]]]

    inputs_to_model = processor(source_image, input_points=input_points, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs_to_model)
    predicted_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs_to_model["original_sizes"], inputs_to_model["reshaped_input_sizes"]
    )
    # We should have 1 mask since we are processing one image with one point
    print(len(predicted_masks))
    print(predicted_masks[0].shape)
    # torch.Size([1, 3, 683, 1024]) means 1 tensor, 3 channels (predictions - default for the model we are using)
    # , 683 image height, 1024 image width
    print(outputs.iou_scores)
    # iou_scores are the confidence score for each mask
    for i in range(3):
        img_utils.show_mask_on_image(source_image, predicted_masks[0][:, i])


def depth_estimation(source_image: Image):
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
    depth.show()




def app_run():
    dinner_image = Image.open("data/jason-briscoe-VBsG1VOgLIU-unsplash.jpg")
    # mask_generation(dinner_image)
    # mask_generation_from_single_point(dinner_image)
    depth_estimation(dinner_image)

app_run()
