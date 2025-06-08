# yolov5/models/mobilenetv2.py

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class MobileNetV2_Backbone(nn.Module):
    def __init__(self, c1, c2, *args):  # YOLOv5는 c1, c2, ... 형태로 args 전달
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.out_channels = c2  # YOLOv5에선 마지막 채널 정보가 필요

    def forward(self, x):
        return self.model(x)