import random
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import BaseDataset
from torch.nn import CrossEntropyLoss,MSELoss
from torch.utils.tensorboard import SummaryWriter
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
        self.datasetDir = "./dataset/val"
        self.batch_size = 8 
        self.num_dim = 59
        self.device = torch.device("cuda")
        self.model_load_path = "checkpoints/last.pth"
        # os.remove(self.tb_log_save_path)

    def init(self):

        self.dataset = BaseDataset(self.datasetDir,True,True)
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=0)
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_dim)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        self.lossfunc  = MSELoss().to(self.device)
        
    def evaluate(self,):
        self.model.eval()
        counter = 0
        total_similarity = 0.0
        total_distance = 0.0
        total_loss = 0.0
        for idx, data in enumerate(self.dataLoader):
            counter+=1
            imgs:torch.Tensor = data["img"]
            labels:torch.Tensor = data["label"]
            imgs=imgs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                output=self.model(imgs)


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