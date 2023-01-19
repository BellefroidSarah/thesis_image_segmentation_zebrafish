from torchvision.ops import masks_to_boxes
from PIL import Image

import torchvision.transforms as transforms
import constants as cst
import numpy as np
import utils as U

import random
import torch
import os


# Dataset class:
# Will use all images in 'img_dir' that have a mask in 'mask_dir' to create a dataset.
# Images and masks have the same name and extention in this project
class ZebrafishDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        """Initialzes a dataset with all the images in the directory img_dir."""
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()

        self.dataset = [file for file in os.listdir(self.imgs) if file in os.listdir(self.masks)]

        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, file))
        return self.transform(img), self.transform(mask), file


# Dataset class implementing K-Fold cross validation.
# Divides all files into K+1 subparts (K=folds)
# Supart K+1 is used for the testing set
# Subpart[actual_fold] is used for the validation set
# other subparts are used for training
class ZebrafishDataset_KFold(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, actual_fold, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
        elif dataset == "test":
            self.dataset = self.files[-1]
            print("Testing set length: {}".format(len(self.dataset)))
        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        img = Image.open(os.path.join(self.imgs, file))
        mask = Image.open(os.path.join(self.masks, file))
        im = self.transform(img)
        img.close()
        ms = self.transform(mask)
        mask.close()
        return im[:3, :, :], ms, file


# Dataset class implementing k-fold cross validation and cropping the image around the head of the fish
# Needs a model trained to segment the fish out of the image. This model will be used to determine where
# the image has to be croppedd
class ZebrafishDataset_KFold_crop_head(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, actual_fold, model, device, dataset="train", folds=5):
        super().__init__()
        self.imgs = img_dir
        self.masks = mask_dir
        # Model trained to segment the fish
        self.fish_model = model
        self.SIZE = (384, 512)
        self.DEVICE = device
        # Transorfms for the fish_model
        self.pretransform = transforms.Compose([transforms.Resize(self.SIZE),
                                                transforms.Pad((0, 64, 0, 64))])
        self.untransform = transforms.Compose([transforms.CenterCrop(self.SIZE),
                                               transforms.Resize((1932, 2576))])
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1

        self.files = [file for file in os.listdir(self.masks) if file in os.listdir(self.imgs)]
        self.files = list(U.split(self.files, self.fold_div))

        # K-fold cross validation
        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
        elif dataset == "test":
            self.dataset = self.files[-1]
            print("Testing set length: {}".format(len(self.dataset)))
        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        img = Image.open(os.path.join(self.imgs, file))
        ms = Image.open(os.path.join(self.masks, file))
        original_image = self.transform(img)
        original_image = original_image[:3, :, :]
        image = self.pretransform(original_image)
        mask = self.transform(ms)

        fish_mask = U.predict_img(self.fish_model, image.unsqueeze(dim=0), self.DEVICE, self.untransform)
        fish_mask_image = Image.fromarray(fish_mask)
        fish_mask_tensor = self.transform(fish_mask_image)

        obj_ids = torch.unique(fish_mask_tensor)
        obj_ids = obj_ids[1:]

        fish_masks = fish_mask_tensor == obj_ids[:, None, None]
        boxes = masks_to_boxes(fish_masks)

        # Coordinates of the fish bounding box
        # Upper left corner coordinates will be used as a starting point to
        # crop the head of the fish
        h_length = boxes[0, 2] + 1 - boxes[0, 0]
        v_length = boxes[0, 3] + 1 - boxes[0, 1]
        h1 = int(boxes[0, 0])
        h2 = int(boxes[0, 2]) + 1
        v1 = int(boxes[0, 1])
        v2 = int(boxes[0, 3]) + 1

        # Needs to be divisible by 5 (it will crop the image to 60% of this size)
        # Making it divisible by two to add same amount of padding
        if h_length % 10 != 0:
            mod = 10 - (h_length % 4)
            h_length += mod

        h_length = (3 * h_length) / 5
        h2 = int(h1 + h_length)

        # Making it divisible by two to add same amount of padding
        if v_length % 2 == 1:
            v1 = v1 - 1
            v_length += 1

        cropped = original_image[:, v1:v2, h1:h2]
        mask = mask[:, v1:v2, h1:h2]

        if h_length > v_length:
            padding = int((h_length - v_length) / 2)
            post_tr = transforms.Compose([transforms.Pad((0, padding, 0, padding)),
                                          transforms.Resize((512, 512))])
        elif h_length > v_length:
            padding = int((v_length - h_length) / 2)
            post_tr = transforms.Compose([transforms.Pad((padding, 0, padding, 0)),
                                          transforms.Resize((512, 512))])
        else:
            post_tr = transforms.Compose([transforms.Resize((512, 512))])

        image = post_tr(cropped)
        mask = post_tr(mask)

        img.close()
        ms.close()
        return image, mask, file, (h_length, v_length)


# Dataset for multi-class segmentation. Implements K-Fold cross validation
class ZebrafishDataset_multi(torch.utils.data.Dataset):
    def __init__(self, actual_fold, dataset="train", folds=5):
        """Initialzes a dataset with all the images in the directory img_dir."""
        super().__init__()
        self.imgs = os.path.join(cst.DIR, "images")
        self.transform = transforms.ToTensor()
        self.fold_div = folds + 1

        self.files = []
        for term in cst.COMBINED_TERM:
            directory = os.path.join(cst.DIR, term)
            for file in os.listdir(directory):
                if "v" in file:
                    if file not in self.files:
                        self.files.append(file)

        self.files = list(U.split(self.files, self.fold_div))

        self.dataset = []
        if dataset == "train":
            for i in range(folds):
                if i != actual_fold:
                    self.dataset = self.dataset + self.files[i]
            print("Training set length: {}".format(len(self.dataset)))
        elif dataset == "validate":
            self.dataset = self.files[actual_fold]
            print("Validation set length: {}".format(len(self.dataset)))
        elif dataset == "test":
            self.dataset = self.files[-1]
            print("Testing set length: {}".format(len(self.dataset)))
            print(self.dataset)
        random.shuffle(self.dataset)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]
        img = Image.open(os.path.join(self.imgs, file))
        shape = (img.height, img.width)
        im = self.transform(img)
        img.close()

        masks = []
        for term in cst.COMBINED_TERM:
            mask_path = os.path.join(cst.DIR, term)
            # The image might not have been annotated but the roi exists. In this case create an
            # empty mask. This is a feature of this particular project.
            # Loss will not be backpropagated on an empty mask.
            if file in os.listdir(mask_path):
                mask = Image.open(os.path.join(mask_path, file))
                ms = self.transform(mask)
                mask.close()
            else:
                mask = np.zeros((shape[0], shape[1]))
                ms = self.transform(mask)
            masks.append(ms)

        m_tuple = tuple(masks)

        annotations = torch.stack(m_tuple, 0)
        annotations = annotations.squeeze(dim=1)

        return im[:3, :, :], annotations, file
