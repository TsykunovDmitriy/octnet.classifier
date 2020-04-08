import torch
from torch import nn
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import os

from dataset import OCTDataset
from utils import *
from logger import Logger
from trainer import Trainer

#augmentation
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

if __name__ == "__main__":
    
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--path_to_train", type=str, default="dataset/OCT2017/train/", help="path to train data")
    parser.add_argument("--path_to_test", type=str, default="dataset/OCT2017/test/", help="path to test data")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--name_of_model", type=str, default="squeezenet1_1", help="name of statdart model")
    opt = parser.parse_args()
    print(opt)

    model = get_model(opt.name_of_model)
    gpu = torch.cuda.is_available()

    if gpu:
        model = model.cuda()
        model = nn.DataParallel(model)

    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights, map_location= "cpu" if not gpu else None))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = F.cross_entropy
    
    augmentation = Compose([
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=0.9)

    data_train = OCTDataset(opt.path_to_train, transforms=augmentation, img_size=(opt.img_size, opt.img_size))
    data_test = OCTDataset(opt.path_to_test, img_size=(opt.img_size, opt.img_size))
    weights = get_weights(data_train)
    sampler_train = WeightedRandomSampler(weights, len(data_train))
    dataloader_train = DataLoader(data_train, batch_size=opt.batch_size, sampler=sampler_train, num_workers=opt.n_cpu)
    dataloader_test = DataLoader(data_test, shuffle=True, batch_size=opt.batch_size, num_workers=opt.n_cpu)

    logger = Logger()

    epochs = 25
    trainer = Trainer(model, 
                    dataloader_train, 
                    dataloader_test, 
                    optimizer, 
                    criterion, 
                    scheduler,  
                    logger,
                    opt.epochs,
                    gpu=gpu,
                    log_interval=10)

    for _ in range(epochs):
        trainer.run_train()
        trainer.run_eval()

