import os.path as osp
import logging
import argparse
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model




def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        (path for key, path in opt['path'].items()
        if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders, test_names = [],[]
    for phase, dataset_opt in opt['datasets'].items():
        for test_name, test_opt in dataset_opt.items():
            test_set = create_dataset(test_opt, 'valid')
            test_loader = create_dataloader(test_set, dataset_opt)
            test_loaders.append(test_loader)
            test_names.append(test_name)
            logger.info('Number of val images in [{:s}]: {:d}'.format( test_name, len(test_set)))


    # create model and load data
    model = create_model(opt)

    for test_name, test_loader in zip(test_names,test_loaders):
        logger.info('\nTesting [{:s}]...'.format(test_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_name)
        util.mkdir(dataset_dir)
        index = 0
        with open(dataset_dir+'/{}.txt'.format(test_name), 'a') as f:
            for data_pair in test_loader:
                model.feed_data(data_pair, Train=False)
                model.test()
                score = model.get_current_score()
                score = float(score.numpy())
                Dist_name = data_pair['Dis_Name'][0].split('/')[-1]
                f.write(Dist_name+','+str(score)+'\n')
                index += 1
                logger.info('Process No.{} Image'.format(index))

if __name__ == '__main__':
    main()