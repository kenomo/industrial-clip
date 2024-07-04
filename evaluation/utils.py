import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np


def convert_box_xyxy_dilate_square(box, image_size=(512, 512), dilation=20, make_squared=True):

    # convert box from xywh to xyxy
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]

    # dilate box
    x1 = x1 - dilation
    y1 = y1 - dilation
    x2 = x2 + dilation
    y2 = y2 + dilation

    # determine max length
    x_diff = x2 - x1
    y_diff = y2 - y1
    max_length = max(x_diff, y_diff)

    # make square
    if make_squared:
        if x_diff < max_length:
            x1 = x1 - (max_length - x_diff) / 2
            x2 = x2 + (max_length - x_diff) / 2
        else:
            y1 = y1 - (max_length - y_diff) / 2
            y2 = y2 + (max_length - y_diff) / 2

    # check if box is out of bounds
    if x1 < 0:
        x1 = 0
        x2 = max(max_length, x2)
    if y1 < 0:
        y1 = 0
        y2 = max(max_length, y2)
    if x2 > image_size[1]:
        x2 = image_size[1]
        x1 = max(0, image_size[1] - max_length)
    if y2 > image_size[0]:
        y2 = image_size[0]
        y1 = max(0, image_size[0] - max_length)

    return (x1, y1, x2, y2)

def segment_image(image, mask):
    segment = np.zeros_like(image)
    segment.fill(255)
    segment[mask] = image[mask]
    return segment

def load_and_resize_image(image_path, max_image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, 
        (int(image.shape[1] / (max(image.shape) / max_image_size)),
         int(image.shape[0] / (max(image.shape) / max_image_size)))
    )
    return (image, resized_image)

def numpy_to_tensor(images, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_tensors = [transform(image)[0:3, :, :] for image in [Image.fromarray(image) for image in images]]
    image_tensors = torch.stack(image_tensors)
    return image_tensors

def get_predictions_above_threshold(res, pred_idx, threshold=0.5):
    return [k for k, idx in enumerate(torch.max(res, dim=1)[1]) if torch.max(res, dim=1)[0][k].item() >= threshold and idx == pred_idx]
