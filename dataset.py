import json
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset



class BaseDataset(Dataset):
    def __init__(self,
                root_dir,
                use_flip=False,
                use_rotation=False,
                im_size:list=(256,256)
                ):
        super(BaseDataset, self).__init__()
        assert isinstance(root_dir, str)
        assert isinstance(use_flip, bool)
        assert isinstance(use_rotation, bool)
        # assert isinstance(is_augment, bool)
        self.root_dir = root_dir
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.is_augment = True if use_flip or use_rotation else False
        self.im_size = im_size

        self.labels:dict = None
        with open(os.path.join(self.root_dir,"labels.json"),'r',encoding="utf-8") as f:
            self.labels:dict = json.load(f)

        self.index = {}
        for idx,key in enumerate(self.labels.keys()):
            self.index[idx] = key


    
    def __getitem__(self, index):
        img_name = self.index[index]
        img_path = os.path.join(self.root_dir,"images",f"{img_name}.png")
        img = self.readImage(img_path)
        label = self.labels[img_name]
        # if img is None:
        #     print("img error: {}".format(img_name))
        # print("img name: {}".format(img_name))
        if self.is_augment:
            img = augment(img, hflip=self.use_flip, rotation=self.use_rotation)


        # BGR to RGB, HWC to CHW, numpy to tensor
        img = img[:, :, [2, 1, 0]]
        if self.im_size is not None:
            img = cv2.resize(img, self.im_size, interpolation = cv2.INTER_AREA)
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,1,0)))).float()
        label=torch.tensor(label)

        return {'img': img_tensor, 'label': label, 'name': img_name}
    

    def readImage(self, filename):
        img=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        # img = cv2.imread(filename=filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img



    def __len__(self):
        return len(self.labels.keys())
    


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): (h, w, c)Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs