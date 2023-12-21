import json
import random
import cv2
import numpy as np
import torch
import torchvision
from core.models.FDENet.FDENet import FDENet
from core.utils.face_data_utils import vectorParse
from core.models.HRNet.hrnet_infer import HRNetInfer

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Extractor(object):
    def __init__(self,):
        self.initConfig()
        self.init()


    def initConfig(self):
        # self.num_dim = 59 -18
        self.num_dim = 7
        self.hidden_dim = 128
        self.device = torch.device("cuda")
        self.model_load_path = "checkpoints/FDE_mouth/epoch 7.pth"
        self.im_size = (256, 256)
        self.hrnet_weight_path:str= "pretrained_models/HR18-WFLW.pth"

    def init(self):
        self.hrnet_infer = HRNetInfer(self.hrnet_weight_path,self.device)
        
        self.model = FDENet(self.num_dim, self.hidden_dim)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        
    def extract(self,filename:str,savepath:str):
        self.model.eval()
        img = self.readImage(filename)
        img = img.to(self.device).unsqueeze(0)
        heatmap:torch.Tensor=self.hrnet_infer.get_heatmap(img)
        heatmap =heatmap.detach()

        with torch.no_grad():
            output=self.model(img,heatmap)
            output = output.squeeze(0).detach().cpu().numpy()
            
            output=self.decode_output(output)

        v=np.zeros(54,dtype=np.float32)
        v[-7:]=output
        output=v
        
        data = vectorParse(output)
        data = json.dumps(data,ensure_ascii=False,indent=4)
        with open(savepath,"w",encoding="utf-8") as f:
            f.write(data)

        return data
    
        
    def decode_output(self,v:torch.Tensor):
        return v*3.0-1.0

    def load_model(self, load_path, model, strict=True):
        load_net = torch.load(load_path)
        model.load_state_dict(load_net, strict=strict)
        model.eval()



    

    def readImage(self, filename):
        img=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        # img = cv2.imread(filename=filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[:, :, [2, 1, 0]]
        if self.im_size is not None:
            img = cv2.resize(img, self.im_size, interpolation = cv2.INTER_AREA)
        # HWC to CHW
        img_np=np.transpose(img, (2,0,1))
        # numpy to tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).float().div(255)
        return img_tensor


if __name__ =="__main__":
    # Step 1, Create an Extractor instance
    extractor = Extractor()
    # Step 2, Extract the face data from image to json file
    data=extractor.extract(filename="test/sutaner_face.png",savepath="test/sutaner_face.json")
    # [Optional] Step 3, Print face data to the console
    print(data)