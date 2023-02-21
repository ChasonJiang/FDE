


import json
import os
import random
import cv2
import numpy as np
from pytools import F
import torch
import torchvision
from face_data_utils import vectorParse
# from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Generator(object):
    def __init__(self,):
        self.datasetDir = "./dataset/val"
        self.num_class = 75
        self.device = torch.device("cuda")
        self.model_load_path = "models/epoch 4.pth"
        self.im_size = (288,512)
        # os.remove(self.tb_log_save_path)


        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_class)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        
    def generate(self,filename:str,savepath:str):
        self.model.eval()
        img = self.readImage(filename)
        img=img.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output=self.model(img)
            output = output.squeeze(0).detach().cpu()
        # print(output)
        # torch.save(output,savepath)
        data = vectorParse(output.numpy())
        data = json.dumps(data,ensure_ascii=False,indent=4)
        with open(savepath,"w",encoding="utf-8") as f:
            f.write(data)

        return data
    
        


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
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,1,0)))).float()
        
        return img_tensor


if __name__ =="__main__":
    generator = Generator()
    # generator.generate("dataset/val/images/upai_chara_0020762.png","test/face_data/upai_chara_0020762.json")
    data=generator.generate("test/images/yuechan_1.jpg","test/face_data/yuechan_1.json")
    # generator.generate("test/images/yuechan_2.jpg","test/face_data/yuechan_2.json")
    # data=generator.generate("test/character_card/xiaowu.png","test/face_data/xiaowu.json")
    print(data)