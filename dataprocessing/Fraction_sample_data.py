"""
script  ver： Aug 19th 17:40
将MIL格式数据集的train抽取一定量部分并命名为AAAA_fraction_XX XX为抽取百分比
"""
import os
import random
import shutil
import argparse
from multiprocessing import Pool, cpu_count


def setup_seed(seed):  # setting up the random seed
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)


def make_and_clear_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


def sampling(file_dir, target_dir, rate, split_subset_range='ALL', CLS=False):
    """
    file_dir: input dataset path
    target_dir: output dataset path
    rate: fraction rate
    split_subset_range:'train' to sample the training only; 'ALL' to sample the training, validation and test sets
    CLS: type of dataset format, True for imagefolder, False for mask+imagefolder format
    """
    print('Dataset at', file_dir)
    split_names = os.listdir(file_dir)
    for split_name in split_names:

        if split_subset_range == 'ALL':
            file_dir_train = os.path.join(file_dir, split_name)
            file_dir_data = os.path.join(file_dir_train, 'data')
            file_dir_mask = os.path.join(file_dir_train, 'mask')
            target_dir_train = os.path.join(target_dir, split_name)
            target_dir_data = os.path.join(target_dir_train, 'data')
            target_dir_mask = os.path.join(target_dir_train, 'mask')

            for type in os.listdir(file_dir_data):

                make_and_clear_path(os.path.join(target_dir_data, type))
                if not CLS:
                    make_and_clear_path(os.path.join(target_dir_mask, type))
                path_dir = os.listdir(os.path.join(file_dir_data, type))  # 取图片的原始路径
                file_number = len(path_dir)
                rate1 = rate  # 自定义抽取的比例（百分制）
                pick_number = int(file_number * rate1 / 100)  # 按照rate比例从文件夹中取一定数量的文件
                sample1 = random.sample(path_dir, pick_number)
                for name in sample1:
                    shutil.copyfile(os.path.join(os.path.join(file_dir_data, type), name),
                                    os.path.join(os.path.join(target_dir_data, type), name))
                    if not CLS:
                        shutil.copyfile(os.path.join(os.path.join(file_dir_mask, type), name),
                                        os.path.join(os.path.join(target_dir_mask, type), name))

        elif split_subset_range == 'train':
            if split_name == 'train':
                file_dir_train = os.path.join(file_dir, split_name)
                file_dir_data = os.path.join(file_dir_train, 'data')
                file_dir_mask = os.path.join(file_dir_train, 'mask')
                target_dir_train = os.path.join(target_dir, split_name)
                target_dir_data = os.path.join(target_dir_train, 'data')
                target_dir_mask = os.path.join(target_dir_train, 'mask')

                for type in os.listdir(file_dir_data):

                    make_and_clear_path(os.path.join(target_dir_data, type))
                    if not CLS:
                        make_and_clear_path(os.path.join(target_dir_mask, type))
                    path_dir = os.listdir(os.path.join(file_dir_data, type))  # 取图片的原始路径
                    file_number = len(path_dir)
                    rate1 = rate  # 自定义抽取的比例（百分制）
                    pick_number = int(file_number * rate1 / 100)  # 按照rate比例从文件夹中取一定数量的文件
                    sample1 = random.sample(path_dir, pick_number)
                    for name in sample1:
                        shutil.copyfile(os.path.join(os.path.join(file_dir_data, type), name),
                                        os.path.join(os.path.join(target_dir_data, type), name))
                        if not CLS:
                            shutil.copyfile(os.path.join(os.path.join(file_dir_mask, type), name),
                                            os.path.join(os.path.join(target_dir_mask, type), name))
            else:
                shutil.copytree(os.path.join(file_dir, split_name), os.path.join(target_dir, split_name))
        else:
            print('not a valid split_list idea')
            raise

        print(split_name, 'has been processed')

    return


def main(args):
    '''
    class_dir = '/Users/munros/Desktop/ROSE_MIL'
    output = r'/Users/munros/Desktop/ROSE/MIL'
    rates = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for rate in rates:

        file_dir = class_dir
        target_dir = os.path.join(output, 'Rose_fraction_' + str(int(rate/10)) + '_MIL')

        sampling(file_dir, target_dir, rate, split_list='train', CLS=False)
    '''
    Dataset_name = os.path.split(args.root)[-1].split('_')[0]
    target_dir = os.path.join(args.save_root, Dataset_name + '_fraction_' + str(int(args.rate / 10)) + '_MIL')

    sampling(args.root, target_dir, args.rate, split_subset_range=args.split_subset_range, CLS=args.CLS)


def get_args_parser():
    parser = argparse.ArgumentParser(description='data_sampling')
    parser.add_argument('--root', default='/root/autodl-tmp/datasets/ROSE_MIL', type=str,
                        help='the data root, not including the final list')
    parser.add_argument('--save_root', default='/root/autodl-tmp/datasets', type=str,
                        help='the data root, not including the final list')
    parser.add_argument('--rate', default=10, type=int,
                        help='the rate of sampling')
    parser.add_argument('--split_subset_range', default='train', type=str,
                        help='the subset which will be sampled: ALL or train')
    parser.add_argument('--CLS', default=False, type=bool,
                        help='the type of dataset: CLS or MIL')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
