import random
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from core.models.FaceNet.FaceNet import EyesNet, FaceNet
from core.dataset.dataset_facenet import BaseDataset
from torch.nn import CrossEntropyLoss,MSELoss
from torch.utils.tensorboard import SummaryWriter

from hrnet import HighResolutionNet
from hrnet_infer import HRNetInfer
from hrnet_utils import decode_preds, get_preds
# from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Evaluator(object):
    def __init__(self,is_init=True):
        if is_init:
            self.initConfig()
            self.init()


    def initConfig(self):
        self.datasetDir = "./dataset/10k/val"
        self.batch_size = 8 
        self.num_dim = 13
        self.device = torch.device("cuda")
        self.model_load_path = "checkpoints/eyes_net/epoch 1.pth"
        self.im_size = (256, 256)
        # os.remove(self.tb_log_save_path)
        self.landmark_model_load_path:str= "pretrained_models/HR18-WFLW.pth"

    def init(self):
        self.init_dataset()
        self.init_model()
        

    def init_dataset(self):

        self.dataset = BaseDataset(self.datasetDir,True,True,False,self.im_size,use_multi_angle=False)
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=0)
        # self.model = FaceNet(self.num_dim)
        # self.landmark_model = HighResolutionNet(is_official=True)
        # # self.load_model(self.landmark_model_load_path,self.landmark_model,False)
        # self.landmark_model.to(self.device).init_weights(self.landmark_model_load_path)
        # self.landmark_model.eval()
        self.hr_infer = HRNetInfer(self.landmark_model_load_path,self.device)

    def init_model(self):
        self.model = EyesNet(98, 128,self.num_dim)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        

    
    def evaluate(self,):
        self.lossfunc  = MSELoss().to(self.device)
        self.model.eval()
        counter = 0
        total_similarity = 0.0
        total_distance = 0.0
        total_loss = 0.0
        for idx, data in enumerate(self.dataLoader):
            counter+=1
            imgs:torch.Tensor = data["img"]
            labels:torch.Tensor = data["label"][:,19:32]
            imgs=imgs.to(self.device)
            # labels:torch.Tensor = data["label"][:,20:32]
            # imgs=imgs.to(self.device)[:,:,35:70,...]
            labels = labels.to(self.device)
            heatmap:torch.Tensor=self.hr_infer.get_heatmap(imgs)
            heatmap =heatmap.detach()

            with torch.no_grad():
                output:torch.Tensor=self.model(heatmap)


            loss=self.lossfunc(output,labels)
            total_loss+=loss
            
            similarity=torch.nn.functional.cosine_similarity(output.detach(),labels.detach(),dim=1)
            similarity = torch.mean(similarity).cpu().numpy()
            distance = torch.nn.functional.pairwise_distance(output.detach(),labels.detach(),p=2).mean()

            total_similarity += similarity
            total_distance+=distance

            print(f"batch: {idx+1} | distance: {distance:.3f} | cosine similarity: {similarity:.3f}")

        print(f"avg loss: {total_loss/counter} | avg similarity: {total_similarity/counter} | avg distance: {total_distance/counter}")
        return total_loss/counter,total_similarity/counter,total_distance/counter


    def load_model(self, load_path, model, strict=True):
        load_net = torch.load(load_path)
        model.load_state_dict(load_net, strict=strict)
        model.eval()



if __name__ =="__main__":
    evaluator = Evaluator()
    evaluator.evaluate()