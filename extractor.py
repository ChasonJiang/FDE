import json
import random
import cv2
import numpy as np
import torch
import torchvision
from face_data_utils import vectorParse
# from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Extractor(object):
    def __init__(self,):
        self.initConfig()
        self.init()


    def initConfig(self):
        # int. Number of feature vector dimensions of face. (Don't modify it, if you don't know what it means)
        self.num_dim = 75
        # to extract on "cpu" or "cuda"(gpu)
        self.device = torch.device("cuda")
        # string.saveing path of model
        self.model_load_path = "models/last.pth"
        # tuple. resize image to the shape. The aspect ratio should be 9:16
        self.im_size = (288,512)


    def init(self):

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_dim)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        
    def extract(self,filename:str,savepath:str):
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
    # Step 1, Create an Extractor instance
    extractor = Extractor()
    # Step 2, Extract the face data from image to json file
    data=extractor.extract(filename="test/images/yuechan_1.jpg",savepath="test/face_data/yuechan_1.json")
    # [Optional] Step 3, Print face data to the console
    print(data)