import os.path as osp
import logging
import time

import options.options as option
import utils.util as util
from data.data_util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model




def main():
    #### options
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = option.parse('/home/jjgu/home/hmcai/MyPyCharm_Pro/PIPAL/Our_IQA/options/test/test_IQA.yml', is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
        if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        print(dataset_opt)
        test_set_PIPAL_Valid = create_dataset(dataset_opt['test_PIPAL_Valid'], 'valid')
        test_set_PIPAL_Full = create_dataset(dataset_opt['test_PIPAL_Full'], 'valid')
        test_set_TID2013 = create_dataset(dataset_opt['test_TID2013'], 'valid')
        # test_set_CSIQ = create_dataset(dataset_opt['test_CSIQ'], 'valid')
        # test_set_LIVE = create_dataset(dataset_opt['test_LIVE'], 'valid')

        test_loader_TID2013 = create_dataloader(test_set_TID2013, dataset_opt)
        test_loader_PIPAL_Full = create_dataloader(test_set_PIPAL_Full, dataset_opt)
        test_loader_PIPAL_Valid = create_dataloader(test_set_PIPAL_Valid, dataset_opt)
        # test_loader_CSIQ = create_dataloader(test_set_CSIQ, dataset_opt)
        # test_loader_LIVE = create_dataloader(test_set_LIVE, dataset_opt)

        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['test_PIPAL_Full']['name'], len(test_set_PIPAL_Full)))
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['test_PIPAL_Valid']['name'], len(test_set_PIPAL_Valid)))
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['test_TID2013']['name'], len(test_set_TID2013)))
        # logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['test_CSIQ']['name'], len(test_set_CSIQ)))
        # logger.info(
        #     'Number of test images in [{:s}]: {:d}'.format(dataset_opt['test_LIVE']['name'], len(test_set_LIVE)))

        test_loaders.append(test_loader_PIPAL_Full)
        test_loaders.append(test_loader_PIPAL_Valid)
        test_loaders.append(test_loader_TID2013)
        # test_loaders.append(test_loader_CSIQ)
        # test_loaders.append(test_loader_LIVE)
    # create model and load data
    model = create_model(opt)
    
    for test_loader in test_loaders:
        
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)
        print("result path is {}".format(dataset_dir))
        index = 0
        with open(dataset_dir+'/{}.txt'.format(test_set_name), 'a') as f:
            for data_pair in test_loader:
                model.feed_data(data_pair, Train=False)
                model.test()
                score = model.get_current_score()
                score = float(score.numpy())
#                Ref_name = data_pair['Ref_path'][0][-9:-4]           
                Dist_name = data_pair['Distortion_path'][0].split('/')[-1]
                f.write(Dist_name+','+str(score)+'\n')
                index += 1
                print(index)
if __name__ == '__main__':
    main()