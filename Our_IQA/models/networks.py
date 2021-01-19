import torch
import logging
import torch.nn as nn
from models.modules.Our_IQA import Our_IQA

logger = logging.getLogger('base')


#### Generator
def define_IQA(opt):
    opt_net = opt['network_G']
    netG = Our_IQA(FENet=opt_net['FENet'], tune=opt_net['tune_flag'], control_bias=opt_net['bias_flag'],
                   real_lpips=opt_net['lpips_flag'])
    return netG


'''Below codes are for network ranker which predict perceptual judgment h from distance pair (d0, d1). 
Find more in Paper <The Unreasonable Effectiveness of Deep Features as a Perceptual Metric> or Github <LPIPS, 
Richard Zhang> '''


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        b, _ = d0.shape
        per = ((judge + 1.) / 2.).view([b, 1, 1, 1])
        d0 = d0.view([b, 1, 1, 1])
        d1 = d1.view([b, 1, 1, 1])
        logit = self.net.forward(d0, d1)
        return self.loss(logit, per)
