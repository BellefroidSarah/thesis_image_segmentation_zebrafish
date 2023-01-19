from unet import UNET

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import torch


# Create segmentation mask
def predict_img(model, image, device, transform, out_threshold=0.5):
    with torch.no_grad():
        x = image
        logits = model(x.to(device))
        logits = transform(logits)
        y_pred = nn.Softmax(dim=1)(logits)
        proba = y_pred.detach().cpu().squeeze(0).numpy()[1, :, :]
        return proba > out_threshold


# Show the images
def show_images(img):
    img = img
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Load a model pre-trained to do binary segmentation
def load_model(filepath):
    net = UNET(3, 2)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net


# Load a model pre-trained to do multi-class segmentation
def load_model_all(filepath):
    net = UNET(3, 14)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net


# Split a list 'a' into 'n' parts, returns a list of 'n' elements,
# each being a sublist of 'a'. If len(a) is not divisible by 'n',
# the sublists will have different lengths.
# For example:
# split([1, 2, 3], 2) = [[1, 2], [3]]
# Source for this function :
# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))