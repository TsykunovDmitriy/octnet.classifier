import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset

class OCTDataset(Dataset):
    def __init__(self, path, transforms=None, img_size=(256, 256)):
        self.transforms = transforms
        self.path = path
        self.img_size = img_size

        classes = [cls_name for cls_name in os.listdir(path) if not cls_name.startswith(".")]
        self.classes = {cls_name: i for i, cls_name in enumerate(classes)}

        self.data = []
        for cls_name in self.classes:
            for name in os.listdir(os.path.join(path, cls_name)):
                if not name.startswith("."):
                    self.data.append(os.path.join(path, cls_name, name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.classes[self.data[idx].split("/")[-2]]

        h, w, _ = img.shape

        if h > w:
            pad_top, pad_bottom = 0, 0
            delta = h - w
            pad_left, pad_right =  delta // 2, delta // 2
        elif w > h:
            pad_left, pad_right = 0, 0
            delta = w - h
            pad_top, pad_bottom =  delta // 2, delta // 2
        else:
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        img = img.transpose(2, 0, 1)
        img = (img / 255.).astype(np.float32)

        img = torch.from_numpy(img)
        
        return img, label