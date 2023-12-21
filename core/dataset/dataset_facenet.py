import json
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BaseDataset(Dataset):
    def __init__(self,
                root_dir,
                use_flip=False,
                use_rotation=False,
                use_colorJitter=False,
                im_size:list=(256,256),
                use_multi_angle=False
                ):
        super(BaseDataset, self).__init__()
        assert isinstance(root_dir, str)
        assert isinstance(use_flip, bool)
        assert isinstance(use_rotation, bool)
        # assert isinstance(is_augment, bool)
        self.root_dir = root_dir
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.use_colorJitter = use_colorJitter
        self.is_augment = True if use_flip or use_rotation else False
        self.im_size = im_size
        self.use_multi_angle=use_multi_angle

        self.labels:dict = None
        with open(os.path.join(self.root_dir,"labels.json"),'r',encoding="utf-8") as f:
            self.labels:dict = json.load(f)

        self.index = {}
        if not self.use_multi_angle:
            for idx,key in enumerate(self.labels.keys()):
                self.index[idx] = key
        else:
            index_ = {}
            for idx,key in enumerate(self.labels.keys()):
                key_ = key.split("_")[0]
                index_[key_] = idx
                
            for idx,key in enumerate(index_.keys()):
                self.index[idx] = key

        self.colorJitter = transforms.ColorJitter(0.5,0.5,0.5,0.5)


    
    def __getitem__(self, index):
        img_name = self.index[index]
        if not self.use_multi_angle:
            img_path = os.path.join(self.root_dir,"images",f"{img_name}.png")
            img = self.readImage(img_path)
            label = self.labels[img_name]
            # if img is None:
            #     print("img error: {}".format(img_name))
            # print("img name: {}".format(img_name))
            img_tensor=self.transform(img)
            
        else:
            img_tensor = []
            label = self.labels[f"{img_name}_0"]
            for i in range(3):
                img_path = os.path.join(self.root_dir,"images",f"{img_name}_{i}.png")
                img = self.readImage(img_path)
                
                # if img is None:
                #     print("img error: {}".format(img_name))
                # print("img name: {}".format(img_name))
                img_tensor.append(self.transform(img))
                
            img_tensor = torch.stack(img_tensor,dim=0)
            
        label=torch.tensor(label,dtype=torch.float32)


        return {'img': img_tensor, 'label': label, 'name': img_name}
    

    def readImage(self, filename):
        img=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        # img = cv2.imread(filename=filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def transform(self,img):
        if self.is_augment:
            img = augment(img, hflip=self.use_flip, rotation=self.use_rotation)


        # BGR to RGB
        # img = img[:, :, [2, 1, 0]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.im_size is not None:
            img = cv2.resize(img, self.im_size, interpolation = cv2.INTER_AREA)
        # img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,1,0)))).float()
        # HWC to CHW
        img_np=np.transpose(img, (2,0,1))
        # numpy to tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).float().div(255)
        if self.use_colorJitter:
            img_tensor=self.colorJitter(img_tensor)
        return img_tensor


    def __len__(self):
        return len(self.index.keys())
    


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