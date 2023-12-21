import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from core.models.FDENet.FDENet import FDENet
from core.dataset.dataset import BaseDataset
from torch.nn import MSELoss

from core.models.HRNet.hrnet_infer import HRNetInfer
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
        self.num_dim = 15
        self.hidden_dim = 128
        self.device = torch.device("cuda")
        self.model_load_path = "checkpoints/FDE_nose/epoch 1.pth"
        self.im_size = (256, 256)
        # os.remove(self.tb_log_save_path)
        self.hrnet_weight_path:str= "pretrained_models/HR18-WFLW.pth"

    def init(self):
        self.hrnet_infer = HRNetInfer(self.hrnet_weight_path,self.device)
        self.dataset = BaseDataset(self.datasetDir,False,False,False,self.im_size)
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False)
        
        self.model = FDENet(self.num_dim, self.hidden_dim)
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
            # labels:torch.Tensor = data["label"]
            imgs=imgs.to(self.device)
            labels:torch.Tensor = data["label"][:, 32:-12]
            labels = labels.to(self.device)
            heatmap:torch.Tensor=self.hrnet_infer.get_heatmap(imgs)
            heatmap =heatmap.detach()

            with torch.no_grad():
                output:torch.Tensor=self.model(imgs,heatmap)


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