import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,downsample:bool=False) -> None:
        super(BasicBlock,self).__init__()
        self.downsample = downsample
        self.block=nn.Sequential(
                            nn.Conv2d(in_channels,out_channels,3,2 if downsample else 1,1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.Conv2d(out_channels,out_channels,3,1,1),
                            nn.BatchNorm2d(out_channels),
                            )
        self.down = nn.Conv2d(in_channels,out_channels,1,2)
        
    def forward(self,x):
        _x=self.block(x)
        if self.downsample:
            return F.relu(self.down(x)+_x)

        return F.relu(x+_x)

        

class FDENet(nn.Module):
    def __init__(self, output_dim:int=59, hidden_dim:int=512) -> None:
        super(FDENet,self).__init__()

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, hidden_dim)     
        self.module = nn.Sequential(
                                BasicBlock(98,128,True), # 64x64 -> 32x32
                                BasicBlock(128,128), # 32x32 -> 32x32
                                BasicBlock(128,256,True), # 32x32 -> 16x16
                                BasicBlock(256,256), # 16x16 -> 16x16
                                BasicBlock(256,512,True), # 16x16 -> 8x8
                                BasicBlock(512,512), # 8x8 -> 8x8
                                nn.Conv2d(512,1024,8),
                                nn.Flatten(),
                                nn.Linear(1024, hidden_dim),
                                )
        
        self.fusion = nn.Sequential(
                                nn.Linear(hidden_dim*2, hidden_dim*4),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim*4, hidden_dim*2),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim*2, hidden_dim),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim, output_dim),
                                )
        

    def forward(self,img:torch.Tensor, heatmap:torch.Tensor):
        residual_feature = self.resnet50(img)
        keypoint_feature = self.module(heatmap)
        feature = torch.cat([residual_feature,keypoint_feature],dim=1)
        o = self.fusion(feature)
        return o


class FDENetPlus(nn.Module):
    def __init__(self, output_dim:dict={"face":19,"eyes":13,"nose":15,"mouth":7}, hidden_dim:int=128) -> None:
        super(FDENetPlus,self).__init__()
        self.face_layer = FDENet(output_dim["face"],hidden_dim=hidden_dim)
        self.eyes_layer = FDENet(output_dim["eyes"],hidden_dim=hidden_dim)
        self.nose_layer = FDENet(output_dim["nose"],hidden_dim=hidden_dim)
        self.mouth_layer = FDENet(output_dim["mouth"],hidden_dim=hidden_dim)

    def forward(self,x:torch.Tensor, heatmap:torch.Tensor):
        face=self.face_layer(x, heatmap)
        eyes=self.eyes_layer(x, heatmap)
        nose=self.nose_layer(x, heatmap)
        mouth=self.mouth_layer(x, heatmap)

        return torch.cat([face,eyes,nose,mouth],dim=1)


class FDENetPlusV2(nn.Module):
    def __init__(self, output_dim:int=59, hidden_dim:int=512) -> None:
        super(FDENet,self).__init__()

        self.layer_1 = torchvision.models.resnet34(pretrained=True)
        self.layer_1.fc = nn.Linear(self.layer_1.fc.in_features, hidden_dim)     

        self.layer_2 = torchvision.models.resnet34(pretrained=True)
        self.layer_2.fc = nn.Linear(self.layer_2.fc.in_features, hidden_dim)     

        self.layer_3 = torchvision.models.resnet34(pretrained=True)
        self.layer_3.fc = nn.Linear(self.layer_3.fc.in_features, hidden_dim)     

        self.layer_4 = torchvision.models.resnet34(pretrained=True)
        self.layer_4.fc = nn.Linear(self.layer_4.fc.in_features, hidden_dim)     
        

        self.kepoint_layer = nn.Sequential(
                                BasicBlock(98,128,True), # 64x64 -> 32x32
                                BasicBlock(128,128), # 32x32 -> 32x32
                                BasicBlock(128,256,True), # 32x32 -> 16x16
                                BasicBlock(256,256), # 16x16 -> 16x16
                                BasicBlock(256,512,True), # 16x16 -> 8x8
                                BasicBlock(512,512), # 8x8 -> 8x8
                                nn.Conv2d(512,1024,8),
                                nn.Flatten(),
                                nn.Linear(1024, hidden_dim),
                                )
        
        self.fusion = nn.Sequential(
                                nn.Linear(hidden_dim*5, hidden_dim*4),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim*4, hidden_dim*2),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim*2, hidden_dim),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_dim, output_dim),
                                )
        

    def forward(self,x:torch.Tensor, heatmap:torch.Tensor):
        feature_1=self.layer_1(x)
        feature_2=self.layer_2(x)
        feature_3=self.layer_3(x)
        feature_4=self.layer_4(x)
        keypoint_feature=self.kepoint_layer(heatmap)
        all_feature=torch.cat([feature_1,
                               feature_2,
                               feature_3,
                               feature_4,
                               keypoint_feature],dim=1)
        o=self.fusion(all_feature)
        return o