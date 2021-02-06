from modules.objects import FieldInfo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import numpy as np
import os
import subprocess
import math
import matplotlib.pyplot as plt
import argparse
import random
import cv2
from pathlib import Path


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn(pretrained=False).features)
        self.layer1 = nn.Sequential(
                        nn.Conv2d(4,64,kernel_size=3,padding=1)
                        )
        self.features = nn.ModuleList(features)[1:]#.eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        return x, results


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 14 x 14
        self.double_conv2 = nn.Sequential(
                        nn.Conv2d(1024,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(256,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 28 x 28
        self.double_conv3 = nn.Sequential(
                        nn.Conv2d(512,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 56 x 56
        self.double_conv4 = nn.Sequential(
                        nn.Conv2d(256,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 112 x 112
        self.double_conv5 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,1,kernel_size=1,padding=0),
                        ) # 256 x 256

    def forward(self,x,concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out) #block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out) #block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out) #block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out) #block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out) #block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)

        # out = torch.sigmoid(out)
        # return out
        return out, torch.sigmoid(out)


class Segmentor(object):
    def __init__(self, feature_weight, relation_weight):

        self.feature_encoder = CNNEncoder().cuda()
        self.relation_network = RelationNetwork().cuda()
        self.feature_encoder.load_state_dict(torch.load(feature_weight, map_location='cuda'))
        self.relation_network.load_state_dict(torch.load(relation_weight, map_location='cuda'))

        self.IMAGE_HEIGHT = 96
        self.IMAGE_WIDTH = 128

    def run(self, field_info: FieldInfo) -> FieldInfo:
        image = field_info.image
        original_size = (image.shape[1], image.shape[0])  # w, h
        image = image[:,:,::-1] # bgr to rgb
        image = image / 255.0
        image = cv2.resize(image, dsize=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        image = np.transpose(image, (2,0,1))
        # calculate features
        query_images_tensor = torch.from_numpy(image).unsqueeze(0).cuda()
        zeros_tensor = torch.zeros(1,1,self.IMAGE_HEIGHT,self.IMAGE_WIDTH, dtype=torch.float, device='cuda')
        query_images_tensor = torch.cat((query_images_tensor,zeros_tensor), dim=1).float().cuda()

        query_features, ft_list = self.feature_encoder(query_images_tensor)
        query_features_ext = query_features
        # relation_pairs = torch.cat((sample_features_ext,query_features_ext),2).view(-1,1024,7,7)
        relation_pairs = torch.cat((query_features, query_features_ext),2).view(-1,1024,3,4)
        output = self.relation_network(relation_pairs,ft_list)
        output_pre_sigmoid, output = output

        output = output.view(-1,1,self.IMAGE_HEIGHT,self.IMAGE_WIDTH)

        # print('='*20)
        # output = (output_pre_sigmoid - output_pre_sigmoid.min()) / (output_pre_sigmoid.max() - output_pre_sigmoid.min())
        # output = output.detach().cpu().numpy()[0][0]
        # output[output < 0.5] = 0
        # output[output >= 0.5] = 255
        # query_pred = output

        query_pred = output.detach().cpu().numpy()[0][0]
        query_pred = np.clip((query_pred*255).astype(np.uint8), 0, 255)
        query_pred = cv2.resize(query_pred, dsize=original_size, interpolation=cv2.INTER_CUBIC)
        query_pred = query_pred.astype(np.uint8)

        cv2.threshold(query_pred, 127, 255, cv2.THRESH_BINARY, dst=query_pred)
        # cv2.medianBlur(query_pred, 5, dst=query_pred)

        field_info.mask_image = query_pred
        field_info.mask_feature = output_pre_sigmoid
        return field_info
