"""
self supervise dataset making    Script  ver： Aug 21th 21:50

todo
提供了一个简单的思路
先做成单进程的，后续做成多进程的


"""
import torch
import numpy as np
import os
import shutil

from utils.tools import to_2tuple, find_all_files


def convert_to_npy(a_data_path):

    # 处理转换

    # 传回npy
    numpy_img = 0

    return numpy_img


def cut_to_patch(numpy_img, save_root, resize_infor, patch_size=384):
    pass


def read_and_convert(data_root,save_root, resize_infor, suffix=None, patch_size=384):
    # 一次处理只一个数据集, 每个数据集的处理方式可能有不同

    # 读入所有数据
    all_files = find_all_files(data_root)

    # 把所有数据转换为同一个格式
    for img in all_files:
        numpy_img = convert_to_npy(img)
        cut_to_patch(numpy_img, save_root, resize_infor, patch_size)

    pass


class to_patch:
    """
    Split a image into patches, each patch with the size of patch_size
    """

    def __init__(self, patch_size=(16, 16)):
        patch_size = to_2tuple(patch_size)
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

    def __call__(self, x):
        c, h, w = x.shape

        assert h // self.patch_h == h / self.patch_h and w // self.patch_w == w / self.patch_w

        num_patches = (h // self.patch_h) * (w // self.patch_w)

        # patch encoding
        # (c, h, w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, self.patch_h, self.patch_w, c)
        # -> (n_patches, patch_size^2*c)
        patches = x.view(
            c,
            h // self.patch_h,
            self.patch_h,
            w // self.patch_w,
            self.patch_w).permute(1, 3, 2, 4, 0).reshape(num_patches, -1)  # it can also used in transformer Encoding

        # patch split
        # (n_patches, patch_size^2*c)
        # -> (num_patches, self.patch_h, self.patch_w, c)
        # -> (num_patches, c, self.patch_h, self.patch_w)
        patches = patches.view(num_patches,
                               self.patch_h,
                               self.patch_w,
                               c).permute(0, 3, 1, 2)

        '''
        # check
        for i in range(len(patches)):
            recons_img = ToPILImage()(patches[i])
            recons_img.save(os.path.join('./patch_play', 'recons_target'+str(i)+'.jpg'))


        # patch compose to image
        # (num_patches, c, self.patch_h, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, c, self.patch_h, self.patch_w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (c, h, w)
        patches = patches.view(h // self.patch_h,
                               w // self.patch_w,
                               c,
                               self.patch_h,
                               self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        '''

        '''
        # visual check
        # reshape
        composed_patches = patches.view(h // self.patch_h,
                                        w // self.patch_w,
                                        c,
                                        self.patch_h,
                                        self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        # view pic
        from torchvision.transforms import ToPILImage
        composed_img = ToPILImage()(bag_image[0])  # transform tensor image to PIL image
        composed_img.save(os.path.join('./', 'composed_img.jpg'))

        '''

        return patches


img = np.ones([3, 224, 224])

patchfy=to_patch(patch_size=(16, 16))

patch=patchfy(img)