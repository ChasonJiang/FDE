import glob
import os
import random
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from core.models.FaceNet.FaceNet import EyesNet, FaceNet
from core.dataset.dataset import BaseDataset
from torch.nn import CrossEntropyLoss,MSELoss,PairwiseDistance
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from eyes_evaluation_facenet import Evaluator
from hrnet_infer import HRNetInfer
import matplotlib.pyplot as plt

# from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Trainer(object):
    def __init__(self,):
        self.initConfig()
        self.init()

    def initConfig(self):
        # string. The root directory of the dataset
        self.datasetDir = "dataset/10k/train"
        self.batch_size = 8 
        # int. Number of feature vector dimensions of eyes
        self.num_dim = 13
        # int. Number of epoch to learn
        self.num_epoch = 20
        # float. learning rate
        self.lr = 0.000005
        # tuple. resize image to the shape 
        # self.im_size = (252, 352)
        self.im_size = (256,256)
        # "cpu" or "cuda"
        self.device = torch.device("cuda")
        # string or None. Load path of model
        self.model_load_path = None
        # string.saveing path of model
        self.model_save_path = "./checkpoints/eyes_facenet"
        # int. Frequency of model saving
        self.save_freq = 1
        # string. Saving path of tensorboard log
        self.tb_log_save_path = "./tb_log/"

        self.landmark_model_load_path:str= "pretrained_models/HR18-WFLW.pth"

    def init(self):
        self.remove_file_in_dir(self.tb_log_save_path)
        self.tb_logger = SummaryWriter(log_dir=self.tb_log_save_path,)
        self.dataset = BaseDataset(self.datasetDir,True,True,True,self.im_size)
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=2)

        # self.landmark_model = HighResolutionNet(is_official=True)
        # # self.load_model(self.landmark_model_load_path,self.landmark_model)
        # self.landmark_model.to(self.device).init_weights(self.landmark_model_load_path)
        # self.landmark_model.eval()
        self.hr_infer = HRNetInfer(self.landmark_model_load_path,self.device)

        self.model = FaceNet(self.num_dim,128)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        self.lossfunc  = MSELoss().to(self.device)
        # self.pairwiseDistance = PairwiseDistance().to(self.device)
        self.optim =torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # self.optim = torch.optim.AdamW()

        # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optim, 1, gamma=0.2, last_epoch=-1)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,self.num_epoch)
        self.evaluator = Evaluator(False)
    
    def remove_file_in_dir(self,dir):
        for file in glob.glob(os.path.join(dir,"*")):  
            os.remove(file)  
            print("Deleted " + str(file)) 


    def train(self,):
        self.model.train()
        step = 0
        for i in range(self.num_epoch):
            for idx, data in enumerate(self.dataLoader):
                step+=1
                self.optim.zero_grad()
                imgs:torch.Tensor = data["img"]
                # labels:torch.Tensor = data["label"][:,19:-5]
                labels:torch.Tensor = data["label"][:,19:32]
                imgs=imgs.to(self.device)
                labels = labels.to(self.device)
                heatmap:torch.Tensor=self.hr_infer.get_heatmap(imgs)
                heatmap =heatmap.detach()
                output:torch.Tensor=self.model(imgs,heatmap)
                

                # loss1=self.lossfunc(output,labels)
                loss1=self.lossfunc(output,labels)
                # loss2=self.pairwiseDistance(output[:,19:-10],labels[:,19:-10]).mean()
                
                # loss = loss1*100
                loss2 = loss1
                # loss = loss1+2.0*loss2
                loss = loss1
                
                loss.backward()
                self.optim.step()

                
                lr = self.get_current_learning_rate()[0]
                similarity=torch.nn.functional.cosine_similarity(output.detach(),labels.detach(),dim=1)
                similarity = torch.mean(similarity)
                distance = torch.nn.functional.pairwise_distance(output.detach(),labels.detach(),p=2).mean()
                self.tb_logger.add_scalar("loss",loss,step)
                self.tb_logger.add_scalar("loss1",loss1,step)
                self.tb_logger.add_scalar("loss2",loss2,step)
                self.tb_logger.add_scalar("lr",lr,step)
                self.tb_logger.add_scalar("batch cosine similarity",similarity,step)
                self.tb_logger.add_scalar("batch distance",distance,step)
                
                

                print(f"epoch: {i+1} | batch: {idx+1} | loss: {loss:.6f} | loss1: {loss1:.6f} | loss2: {loss2:.6f} | lr: {lr} | distance: {distance:.3f} | cosine similarity: {similarity.cpu().numpy():.3f}")

            if (i+1)%self.save_freq==0:
                self.save_model(self.model,f"epoch {i+1}")
            self.val(i+1)
            self.scheduler.step()
            
        self.save_model(self.model,f"last")
        
    def val(self,epoch):
        self.evaluator.initConfig()
        self.evaluator.model_load_path = os.path.join(self.model_save_path,f"epoch {str(epoch)}.pth")
        self.evaluator.init()
        avg_loss,avg_similarity,avg_distance=self.evaluator.evaluate()
        self.tb_logger.add_scalar("val avg loss",avg_loss,epoch)
        self.tb_logger.add_scalar("val avg similarity",avg_similarity,epoch)
        self.tb_logger.add_scalar("val avg distance",avg_distance,epoch)
        

    def get_current_learning_rate(self):
        lr_l = []
        for param_group in self.optim.param_groups:
            lr_l.append(param_group['lr'])
        return lr_l
    
    # def cosine_similarity(x,y):
    #     num = x.dot(y.T)
    #     denom = np.linalg.norm(x) * np.linalg.norm(y)
    #     return num / denom


    def save_model(self, model, name):
        
        save_filename = '{}.pth'.format(name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        save_path = os.path.join(self.model_save_path, save_filename)
        print('Saving model [{:s}] ...'.format(save_path))
        state_dict = model.state_dict()
        # for key, param in state_dict.items():
        #     state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # def load_model(self, load_path:str, model:torch.nn.Module, strict=False):
    #     # load_net = torch.load(load_path)
    
    #     # model.load_state_dict(load_net, strict=strict)
    #     # model.eval()
    #     pretrained_dict = torch.load(load_path)
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                         if k in model_dict.keys()}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     model.eval()


    def load_model(self, load_path, model, strict=False):
        load_net = torch.load(load_path)
        model.load_state_dict(load_net, strict=strict)
        model.eval()



if __name__ =="__main__":
    trainer = Trainer()
    trainer.train()