import json
import random
import cv2
import numpy as np
import torch
import torchvision
from core.models.FaceNet.FaceNet import EyesNet, FaceNet
from face_data_utils import vectorParse
from hrnet import HighResolutionNet
from hrnet_infer import HRNetInfer
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
        self.num_dim = 13
        # to extract on "cpu" or "cuda"(gpu)
        self.device = torch.device("cuda")
        # string.saveing path of model
        self.model_load_path = "checkpoints/eyes_facenet/epoch 13.pth"
        # self.model_load_path = "models/last_head_1_34_30_with_colorjitter.pth"
        # tuple. resize image to the shape. The aspect ratio should be 9:16
        # self.im_size = (252, 352)
        self.im_size = (256, 256)
        self.landmark_model_load_path:str= "pretrained_models/HR18-WFLW.pth"



    def init(self):
        self.hr_infer = HRNetInfer(self.landmark_model_load_path,self.device)
        self.model = FaceNet(self.num_dim,128)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        
        
    def extract(self,filename:str,savepath:str):
        self.model.eval()
        img = self.readImage(filename)
        img = img.to(self.device).unsqueeze(0)
        heatmap:torch.Tensor=self.hr_infer.get_heatmap(img)
        heatmap =heatmap.detach()

        # imgs_zeros = torch.zeros_like(img)
        # imgs_zeros[:,:,35:70,...] = img[:,:,35:70,...]
        # img = imgs_zeros
        
        # img:torch.Tensor = img[:,:,35:70,...]
        # img = torchvision.transforms.Resize((128,128))(img)
        with torch.no_grad():
            output=self.model(img,heatmap)
            output = output.squeeze(0).detach().cpu().numpy()
            
            output=self.decode_output(output)
        # print(output)
        # torch.save(output,savepath)

        v=np.zeros(54,dtype=np.float32)
        v[19:32]=output
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