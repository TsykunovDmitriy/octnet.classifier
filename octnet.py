import torch

import cv2
import numpy as np

from utils import get_model

class OCTnet:
    def __init__(self, name_of_model="squeezenet1_1", path_to_weights="checkpoints/squeezenet_accuracy_99.700.pth", img_size=(256, 256), max_batch_size=8):

        self.model = get_model(name_of_model)

        self.gpu = torch.cuda.is_available()
        self.classes = {0: 'DRUSEN', 1: 'CNV', 2: 'DME', 3: 'NORMAL'}

        if self.gpu:
            self.model = self.model.cuda()

        self.model.load_state_dict(torch.load(path_to_weights, map_location="cpu" if not self.gpu else None))
        self.model.eval()

        self.img_size = img_size
        self.max_batch_size = max_batch_size
    
    def __call__(self, imgs):
        # preprocessing
        imgs_ = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) for img in imgs]

        for i in range(len(imgs_)):
            h, w, _ = imgs_[i].shape

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

            img = cv2.copyMakeBorder(imgs_[i], pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)

            imgs_[i] = img
            
        imgs_ = [(img.transpose(2, 0, 1) / 255.).astype(np.float32) for img in imgs_]
        imgs_ = [torch.from_numpy(img).unsqueeze(0) for img in imgs_]
        imgs_ = torch.cat(imgs_)

        # CUDA
        if self.gpu:
            imgs_ = imgs_.cuda()

        #inference
        predicts = []
        with torch.no_grad():
            if len(imgs_) < self.max_batch_size:
                predicts = self.model(imgs_)
                _, predicts = torch.max(predicts.data, 1)
                final_predicts = list(predicts.cpu().data.numpy())
            else:
                predicts = []
                for i in range(len(imgs_) // self.max_batch_size):
                    pred = self.model(imgs_[i*self.max_batch_size : i*self.max_batch_size + self.max_batch_size])
                    _, pred = torch.max(pred.data, 1)
                    predicts.append(pred)
                else:
                    if (len(imgs_) % self.max_batch_size) != 0:
                        pred = self.model(imgs_[i*self.max_batch_size + self.max_batch_size : ])
                        _, pred = torch.max(pred.data, 1)
                        predicts.append(pred)
                final_predicts = []
                for pr in predicts:
                    final_predicts += list(pr.cpu().data.numpy())
        
        final_predicts = [self.classes[pr] for pr in final_predicts]

        return final_predicts