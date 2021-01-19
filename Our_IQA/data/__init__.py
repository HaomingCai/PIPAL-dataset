'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
                                           pin_memory=True)


def create_dataset(dataset_opt, mode):
    dataset = dataset_opt['name']  # BAPPS or PIPAL

    if mode == 'train':
        if dataset == 'PIPAL':
            from data.PairedTrain_dataset import PIPALDataset as D
        elif dataset == 'BAPPS':
            from data.PairedTrain_dataset import BAPPSDataset as D
        else:
            raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

        dataset = D(dataset_opt)
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
        return dataset

    elif mode == 'valid':
        if dataset == 'PIPAL' or 'TID2013':
            from data.ValidorTest_dataset import ValidDataset as D
        else:
            raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
        dataset = D(dataset_opt)
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
        return dataset

    else:
        raise NotImplementedError(' Dataset is not recognized.')

