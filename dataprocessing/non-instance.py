"""
对测试集进行处理，将图片中的实例遮住，实例部分用原图像素均值填充，生成新的测试集
ver： Feb 21th
"""

import numpy as np
import cv2
import os


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


if __name__ == '__main__':
    # 只需要修改数据路径和result路径，new_test与test平级
    # 导入测试集image和mask
    data_path = 'E:/Study/code/datasets/SIPaKMeD_MIL/test/data/'  # MIL数据集的路径
    result_path = 'E:/Study/code/datasets/SIPaKMeD_MIL/new_test/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    suffix = '.jpg'
    # 获取类别名，制作label和类别名的对应字典
    class_names = [filename for filename in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, filename))]
    class_names.sort()
    cls_idxs = [i for i in range(len(class_names))]
    class_id_dict = dict(zip(class_names, cls_idxs))
    input_ids = sorted(find_all_files(data_path, suffix=suffix))

    # 制作结果路径
    for class_name in class_names:
        res_path = result_path + class_name
        if not os.path.exists(res_path):
            os.makedirs(res_path)

    for i in range(len(input_ids)):
        image_path = input_ids[i]
        # 读取image和mask
        # CV2 0-255 hwc，in totensor step it will be transformed to chw.  ps:PIL(0-1 hwc)
        image = np.array(cv2.imread(image_path))

        # mask_path is replace the last 'data' by 'mask'
        mask_path = "data".join(image_path.split("data")[:-1]) + 'mask' + "".join(image_path.split("data")[-1:])
        # mask: 0/255 cv2 hwc
        mask = np.array(cv2.imread(mask_path))
        mask_norm = np.where(mask > 50, 0, 1)
        # new_image_path is replace the last 'test/data' by 'new_test'
        new_image_path = "data".join(image_path.split("test/data")[:-1]) + \
                         'new_test' + "".join(image_path.split("test/data")[-1:])
        new_image = image * mask_norm

        # 把抠掉的部分填充成原图的像素均值
        value_mean_r = int(np.mean(image[:, :, 0]))
        value_mean_g = int(np.mean(image[:, :, 1]))
        value_mean_b = int(np.mean(image[:, :, 2]))
        new_image[:, :, 0][new_image[:, :, 0] == 0] = value_mean_r
        new_image[:, :, 1][new_image[:, :, 1] == 0] = value_mean_g
        new_image[:, :, 2][new_image[:, :, 2] == 0] = value_mean_b
        new_image = new_image.astype(np.uint8)

        # # 显示原图，mask，new_image
        # images = np.hstack([image, mask, new_image])
        # cv2.imshow('Before and after mask', images)
        # cv2.waitKey(0)

        # 存储新的图片
        cv2.imwrite(new_image_path, new_image)

