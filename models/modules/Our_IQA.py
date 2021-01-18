import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import time
import torch.nn.parallel


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), nn.ReLU(), ]
        self.model = nn.Sequential(*layers)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class Our_IQA(nn.Module):
    def __init__(self, FENet='Alex', tune=False, control_bias=True, real_lpips=False):

        super(Our_IQA, self).__init__()

        self.FENet = FENet
        self.tune = tune
        self.c_bias = control_bias
        self.real_lpips = real_lpips

        #### Init All Components
        self.init_feature_extraction()
        self.init_score_regression()
        self.cuda()

    def init_feature_extraction(self):
        if self.FENet == 'VGG16':
            self.FE = models.vgg16(pretrained=True).features
            self.fea_dims = [64, 128, 256, 512, 512]
            self.ddim = 5
            # self.hooks = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 29)]  # Before Activation
            self.hooks = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]  # Before MaxPool, After Activation
            # self.hooks = [(0, 5), (5, 10), (10, 17), (17, 24), (24, 31)] # After MaxPool
        elif self.FENet == 'Alex':
            self.FE = models.alexnet(pretrained=True).features
            self.fea_dims = [64, 192, 384, 256, 256]
            self.ddim = 5
            # self.hooks = [(0, 1), (1, 4), (4, 7), (7, 9), (9, 11)]  # Before Activation
            self.hooks = [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)]  # Before MaxPool
            # self.hooks = [(0, 3), (3, 6), (6, 8), (8, 10), (10, 13)] # After MaxPool
        else:
            raise Exception('Wrong Feature Extractor')

        for param in self.FE.parameters():
            param.requires_grad = self.tune

    def cuda(self):
        self.FE.cuda()
        if self.real_lpips:
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()
        else:
            self.regreassion_mine_1.cuda()
            self.regreassion_mine_2.cuda()
            self.regreassion_mine_3.cuda()
            self.regreassion_mine_4.cuda()
            self.regreassion_mine_4.cuda()

    def init_score_regression(self):
        if (self.real_lpips):
            self.lin0 = NetLinLayer(self.fea_dims[0], use_dropout=True)
            self.lin1 = NetLinLayer(self.fea_dims[1], use_dropout=True)
            self.lin2 = NetLinLayer(self.fea_dims[2], use_dropout=True)
            self.lin3 = NetLinLayer(self.fea_dims[3], use_dropout=True)
            self.lin4 = NetLinLayer(self.fea_dims[4], use_dropout=True)
        else:
            self.regression_models = []
            self.regreassion_mine_1 = nn.Sequential(
                nn.Conv2d(self.fea_dims[0], self.fea_dims[0], 1, bias=self.c_bias),
                nn.ReLU(),
                nn.Conv2d(self.fea_dims[0], 1, 1, bias=self.c_bias),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)))

            self.regreassion_mine_2 = nn.Sequential(
                nn.Conv2d(self.fea_dims[1], self.fea_dims[1], 1, bias=self.c_bias),
                nn.ReLU(),
                nn.Conv2d(self.fea_dims[1], 1, 1, bias=self.c_bias),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)))

            self.regreassion_mine_3 = nn.Sequential(
                nn.Conv2d(self.fea_dims[2], self.fea_dims[2], 1, bias=self.c_bias),
                nn.ReLU(),
                nn.Conv2d(self.fea_dims[2], 1, 1, bias=self.c_bias),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)))

            self.regreassion_mine_4 = nn.Sequential(
                nn.Conv2d(self.fea_dims[3], self.fea_dims[3], 1, bias=self.c_bias),
                nn.ReLU(),
                nn.Conv2d(self.fea_dims[3], 1, 1, bias=self.c_bias),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)))

            self.regreassion_mine_5 = nn.Sequential(
                nn.Conv2d(self.fea_dims[4], self.fea_dims[4], 1, bias=self.c_bias),
                nn.ReLU(),
                nn.Conv2d(self.fea_dims[4], 1, 1, bias=self.c_bias),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)))

    def compute_features(self, img_input):
        fea_list = []
        fea_list.append(img_input)
        for i in range(self.ddim):
            fea_list.append(
                nn.Sequential(*list(self.FE.children())[self.hooks[i][0]: self.hooks[i][1]])(fea_list[-1])
            )
        return fea_list

    def compute_score(self, diff_list):
        if self.real_lpips:
            score_list = []
            assert len(diff_list) == self.ddim, "Check compute score"
            score_list.append(spatial_average(self.lin0.model(diff_list[0]), keepdim=True))
            score_list.append(spatial_average(self.lin1.model(diff_list[1]), keepdim=True))
            score_list.append(spatial_average(self.lin2.model(diff_list[2]), keepdim=True))
            score_list.append(spatial_average(self.lin3.model(diff_list[3]), keepdim=True))
            score_list.append(spatial_average(self.lin4.model(diff_list[4]), keepdim=True))
            return score_list
        else:
            score_list = []
            assert len(diff_list) == self.ddim, "Check compute score"
            score_list.append(self.regreassion_mine_1(diff_list[0]))
            score_list.append(self.regreassion_mine_2(diff_list[1]))
            score_list.append(self.regreassion_mine_3(diff_list[2]))
            score_list.append(self.regreassion_mine_4(diff_list[3]))
            score_list.append(self.regreassion_mine_5(diff_list[4]))
            return score_list

    def forward(self, image_ref, image_A):
        '''Extract Feature A and Feature Ref'''
        fea_ref = self.compute_features(image_ref)
        fea_A = self.compute_features(image_A)

        ''' Feature Ref - Feature A '''
        diff_list = []
        for i in range(self.ddim):
            diff_list.append(fea_ref[i + 1] - fea_A[i + 1])

        '''Regression of Subtraction of  Fea Ref - Fea A '''
        score_fea_list = self.compute_score(diff_list)

        '''Average'''
        score_fea = torch.cat(score_fea_list, 1)
        batch_size, score_size, _, _ = score_fea.shape
        score_fea = score_fea.view(batch_size, score_size)
        score = score_fea.sum(1).view(batch_size, -1)

        return score
