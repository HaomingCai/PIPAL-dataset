import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class IQA_Model(BaseModel):
    def __init__(self, opt):
        super(IQA_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netIQA = networks.define_IQA(opt).to(self.device)
        self.rankLoss = networks.BCERankingLoss().to(self.device)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netIQA.train()

            # weight of loss
            self.l_rank_w = train_opt['BCE_weight']

            # optimizers
            optim_params = []
            wd_G = train_opt['weight_decay_IQA'] if train_opt['weight_decay_IQA'] else 0
            for k, v in self.netIQA.named_parameters():
                optim_params.append(v)
            for k, v in self.rankLoss.named_parameters():
                optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_IQA'], weight_decay=wd_G,
                                                betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_G)
            self.log_dict = OrderedDict()

    def feed_data(self, data, Train=True):
        if Train:
            self.Ref = data['Ref'].to(self.device)
            self.Distortion_A = data['Dist_A'].to(self.device)
            self.Distortion_B = data['Dist_B'].to(self.device)
            self.probability_AB = data['probability_AB'].to(self.device)
            self.tensor_one = torch.tensor([1]).float().to(self.device)
        else:
            self.Ref = data['Ref'].to(self.device)
            self.Distortion = data['Distortion'].to(self.device)

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        # Obtain Objective Score
        predict_pro_Ref_A = self.netIQA(self.Ref, self.Distortion_A).float()
        predict_pro_Ref_B = self.netIQA(self.Ref, self.Distortion_B).float()

        # Predict perceptual judgment h from distance pair (d0, d1). See Paper of LPIPS.
        B, _ = predict_pro_Ref_A.shape
        var_judge = self.probability_AB.view(predict_pro_Ref_A.size())
        self.loss = self.l_rank_w * (
            self.rankLoss.forward(predict_pro_Ref_A, predict_pro_Ref_B, var_judge * 2. - 1.)
        )

        # Optimization
        self.clamp_weights()
        self.loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['loss_total'] = self.loss.item()

    def test(self):
        self.netIQA.eval()
        with torch.no_grad():
            self.IQA_score = self.netIQA(self.Ref, self.Distortion)
        self.netIQA.train()

    def clamp_weights(self):
        for module in self.netIQA.modules():
            try:
                if (hasattr(module, 'weight') and module.kernel_size == (1, 1)):
                    module.weight.data = torch.clamp(module.weight.data, min=0)
            except:
                pass

    def get_current_log(self):
        return self.log_dict

    def get_current_score(self):
        return self.IQA_score.detach().float().cpu()

    def print_network(self):
        s, n = self.get_network_description(self.netIQA)
        if isinstance(self.netIQA, nn.DataParallel) or isinstance(self.netIQA, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netIQA.__class__.__name__,
                                             self.netIQA.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netIQA.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        try:
            load_path_G = self.opt['path']['pretrain_model_G']
            load_path_R = self.opt['path']['pretrain_model_R']
        except:
            load_path_G, load_path_R = None, None

        if (load_path_R is not None):
            logger.info('Loading model for R [{:s}] ...'.format(load_path_R))
            self.load_network(load_path_R, self.rankLoss, self.opt['path']['strict_load'])
        if (load_path_G is not None):
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netIQA, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netIQA, 'G', iter_label)
        self.save_network(self.rankLoss, 'R', iter_label)
