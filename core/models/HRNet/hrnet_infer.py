import os
import argparse
import time
import cv2
import torch
import sys

import numpy as np
import onnxruntime
from core.models.HRNet.hrnet import HighResolutionNet
from core.models.HRNet.hrnet_utils import decode_preds

class HRNetInfer():
    def __init__(self,weight_path:str,device:torch.device=torch.device("cpu")) -> None:
        self.model = HighResolutionNet(is_official=True)
        self.model.to(device).init_weights(weight_path)
        self.model.eval() 
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            self.face_model_ort = onnxruntime.InferenceSession(os.path.join(os.path.abspath(os.path.dirname(__file__)),self.face_weight_path)
                                                               ,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', "ROCMExecutionProvider", "DmlExecutionProvider",'CPUExecutionProvider'])
                                                                #  ,providers=['DmlExecutionProvider', 'CUDAExecutionProvider'])
        except:
            raise Exception("Face模型加载失败！")

        
    def get_heatmap(self, imgs:torch.Tensor):
        with torch.no_grad():
            heatmap = self.model(imgs)
        return heatmap
    
    def infer_from_numpy(self,np_img:np.ndarray):
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        np_img = cv2.resize(np_img, [256,256])
        input = np.copy(np_img).astype(np.float32)
        input = (input / 255.0 - self.mean) / self.std
        input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0).to(self.device)

        return self.infer(input)
    
    def infer(self,imgs:torch.Tensor):

        h=imgs.size(2)
        w=imgs.size(3)
        heatmap = self.get_heatmap(imgs)
        predes=decode_preds(heatmap, torch.tensor([[h/2.0,w/2.0] for i in range(int(heatmap.size(0)))]),\
                            torch.tensor([h/200.0 for i in range(int(heatmap.size(0)))]), [int(h/4),int(w/4)]).detach()                
        landmark_img=self.prede_to_img(predes,[h,w])

        return landmark_img,predes

    def prede_to_img(self,predes,size:tuple):
        '''
            predes shape:[b,98,2]
        '''
        img=np.zeros((predes.size(0),size[0],size[1],1))
        

        for batch_idx in range(predes.size(0)):
            for x,y in predes[batch_idx]:    
                x, y = int(x), int(y)
                img[batch_idx]=cv2.circle(img[batch_idx], (x,y), 1, (1), 2)
                
        # img=torchvision.transforms.Resize(size)(torch.tensor(img))
        # return img
        return torch.tensor(img).permute(0,3,1,2).to(self.device).float()


    
if __name__ == '__main__':
    img_path=r"dataset\\10k\\train\\images\\ffc02951-1e29-11ee-a160-d4d853743ca5.png"
    weight_pth = "pretrained_models\HR18-WFLW.pth"
    device = torch.device("cuda")
    hr_infer=HRNetInfer(weight_pth, device)
    img=cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    hr_infer.infer_from_numpy(img)
