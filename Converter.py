import numpy as np
import torch
import torchvision
import onnxruntime
from core.models.FDENet.FDENet import FDENet

from core.models.HRNet.hrnet import HighResolutionNet

class Converter():
    def __init__(self,):
        self.initConfig()
        self.init()


    def initConfig(self):
        self.model_name = "weights/FDENet_nose.onnx"
        # int. Number of feature vector dimensions of face. (Don't modify it, if you don't know what it means)
        self.num_dim = 15
        self.hidden_dim = 128
        # to extract on "cpu" or "cuda"(gpu)
        self.device = torch.device("cpu")
        # string.saveing path of model
        self.model_load_path = "checkpoints\FDE_nose\last.pth"
        # tuple. resize image to the shape. The aspect ratio should be 9:16
        # self.im_size = (252, 352)
        # self.input_shape = [1,3,256,256]
        # self.input_shape = [1,3,-1,-1]


    def init(self):

        # self.model = torchvision.models.resnet50(pretrained=True)
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_dim)
        # if self.model_load_path is not None:
        #     self.load_model(self.model_load_path,self.model,False)
        # self.model=self.model.to(self.device)
        # self.model = HighResolutionNet(self.num_dim,pool="max")
        # self.model = HighResolutionNet(is_official=True)
        self.model = FDENet(self.num_dim, self.hidden_dim)
        if self.model_load_path is not None:
            self.load_model(self.model_load_path,self.model)
        self.model=self.model.to(self.device)
        
        
    def load_model(self, load_path, model, strict=False):
        load_net = torch.load(load_path)
        model.load_state_dict(load_net, strict=strict)
        model.eval()
        
    def convert(self,):
        # tensor_shape = [-i*10 if i==-1 else i for i in self.input_shape]
        image = torch.zeros([1,3,256,256])
        heatmap = torch.zeros([1,98,64,64])
        dynamic_axes=[1,3,-1,-1]
        # for idx,item in enumerate(self.input_shape):
        #     if item==-1:
        #         dynamic_axes.append(idx)
                
        torch.onnx.export(self.model,
                        (image,heatmap),
                        self.model_name,
                        input_names=["image", "heatmap"],
                        output_names=["out"],
                        # dynamic_axes={"image":dynamic_axes,
                        #               "heatmap":[1,98,-1,-1]
                        #               }
                        )


if __name__ =="__main__":
    converter = Converter()
    converter.convert()
    
    ort_session = onnxruntime.InferenceSession("weights/FDENet_nose.onnx")
    image = np.zeros([1,3,256,256],dtype=np.float32)
    heatmap = np.zeros([1,98,64,64],dtype=np.float32)
    output=ort_session.run(['out'], {'image': image,"heatmap":heatmap})
    print(output)