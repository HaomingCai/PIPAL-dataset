import os
import math
import torch
import random
import logging
import argparse
import pandas as pd
from utils import util
from models import create_model
import torch.distributed as dist
import options.options as option
import torch.multiprocessing as mp
from data import create_dataloader, create_dataset




def obtain_MOS_Score(score_root):
    score_list = []
    fnames = [fname for fname in os.listdir(score_root) if '.txt' in fname]
    for fname in sorted(fnames):
        ELO_path = os.path.join(score_root, fname)
        with open(ELO_path, 'r') as f:
            lines = f.readlines()
            score_list += [float(line.split(',')[1][:-1]) for line in lines]
    return score_list


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    os.environ['RANK'] = '0'
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()


    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None


    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model_G' not in key and 'pretrain_model_R' not in key and 'resume' not in key and 'strict_load' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')


    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)


    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True


    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_choice = dataset_opt['choice']
            train_sampler = None
            if dataset_choice == 'BAPPS':
                train_BAPPS_set = create_dataset(dataset_opt['train_BAPPS'], mode='train')
                train_size = int( math.ceil( len(train_BAPPS_set) )  / (dataset_opt['train_BAPPS']['batch_size']))
                train_BAPPS_loader = create_dataloader(train_BAPPS_set, dataset_opt['train_BAPPS'], opt, train_sampler)
                assert train_BAPPS_loader is not None
            elif dataset_choice == 'PIPAL':
                train_PIPAL_set = create_dataset(dataset_opt['train_PIPAL'], mode='train')
                train_size = int( math.ceil( len(train_PIPAL_set) )  / (dataset_opt['train_PIPAL']['batch_size']))
                train_PIPAL_loader = create_dataloader(train_PIPAL_set, dataset_opt['train_PIPAL'], opt, train_sampler)
                assert train_PIPAL_loader is not None
            else:
                raise NotImplementedError('Chosen Training Dataset is not recognized. Only support PIPAL or BAPPS')

            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))

            if rank <= 0:
                if dataset_choice == 'BAPPS':
                    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_BAPPS_set), train_size))
                elif dataset_choice == 'PIPAL':
                    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_PIPAL_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
            else:
                raise NotImplementedError(' choice of datasets is not recofnized. ')


        elif phase == 'val':
            valid_loaders, valid_MOSs, valid_names = [], [], []
            valid_opts = dataset_opt
            for valid_name, valid_opt in valid_opts.items():
                valid_MOS = valid_opt['mos_root']
                valid_set = create_dataset(valid_opt, 'valid')
                valid_loader = create_dataloader(valid_set, dataset_opt)
                valid_loaders.append(valid_loader)
                valid_MOSs.append(valid_MOS)
                valid_names.append(valid_name)
                if rank <= 0:
                    logger.info('Number of val images in [{:s}]: {:d}'.format(
                        valid_name, len(valid_set))
                    )
                assert valid_set is not None
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))



    #### create model
    model = create_model(opt) 

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format( resume_state['epoch'], resume_state['iter']) )
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        if dataset_choice == 'BAPPS':
            dataloader_list = [train_BAPPS_loader]
        elif dataset_choice == 'PIPAL':
            dataloader_list = [train_PIPAL_loader]
        else:
            raise NotImplementedError('Your dataset choice ({}) is not recognized.'.format(dataset_choice))


        for train_loader in dataloader_list:
            for train_data in train_loader:
                current_step += 1
                if current_step > total_iters:
                    break

                model.feed_data(train_data)
                model.optimize_parameters(current_step)

                #### log
                if current_step % opt['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, current_step, model.get_current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)


                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)


                # Validation
                if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                    for valid_name, val_mos_root, val_loader in zip(valid_names, valid_MOSs, valid_loaders):
                        index = 0
                        MOS_List = obtain_MOS_Score(val_mos_root)
                        IQA_List = []
                        txt_Fname = os.path.join(opt['path']['val_images'], "{}_{}_iter{}.txt".format(opt['name'], valid_name, current_step))

                        logger.info('Validation Testing, Please Wait')
                        with open(txt_Fname, 'a') as f:
                            for val_data in val_loader:
                                index += 1
                                if index % 1000 == 0:
                                    logger.info("Processing No.{} Image in PIPAL_Full".format(index))
                                model.feed_data(val_data, Train=False)
                                model.test()
                                score = model.get_current_score()
                                score = float(score.numpy())
                                Dist_name = val_data['Dis_Name'][0].split('/')[-1]
                                f.write(Dist_name + ',' + str(score) + '\n')
                                IQA_List.append(score)

                        # Calculate Correlation between MOS and IQA scores
                        IQA_list_pd = pd.Series(IQA_List)
                        MOS_list_pd = pd.Series(MOS_List)
                        SROCC =  MOS_list_pd.corr(IQA_list_pd, method='spearman')

                        # Record corr on Tensorboard
                        logger.info('# Validation # {}_SROCC: {:.4e}'.format(valid_name,SROCC))
                        tb_logger.add_scalar('{}_SROCC'.format(valid_name), SROCC, current_step)


    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')



if __name__ == '__main__':
    main()
