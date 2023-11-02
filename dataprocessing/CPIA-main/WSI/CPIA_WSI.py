"""
CPIA_WSI.py  ver： 23 Nov 2
This code aims to split each whole slide image into standardised patches.
The patch sizes are: 3840, 960, 384, 96
"""
import os
import PIL.Image as Image
import numpy as np
import openslide
import torch
from PIL import ImageFile
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

STANDARD_MPP = 0.4942
patch_size = [(3840, 3840), (960, 960), (384, 384), (96, 96)]


def run_map_mp(func, argument_list, num_processes='', is_tqdm=True):
    """Multi-threading with progress bar
    Ref: https://zhuanlan.zhihu.com/p/359369130

    Args:
        func (func): Target function.
        argument_list (list): Argument list. Example format: [(a1, b1), (a2, b2)]
        num_processes (str, optional): The number of processes. Defaults to the number of threads - 3.
        is_tqdm (bool, optional): Whether to display progress bar (using tqdm). Defaults to True.

    Returns:
        result_list_tqdm: Output list of each thread.
    """
    result_list_tqdm = []
    try:
        if not num_processes:
            num_processes = min(cpu_count() - 3, len(argument_list))
        pool = Pool(processes=num_processes)
        print('start running multiprocess using {} threads'.format(num_processes))

        # Use pool.starmap to allow multi-parameter for func
        if is_tqdm:

            # Here because starmap can only return result after fully finished, it is not capable to
            # operate with tqdm progress bar.

            # In this case, I update the progress bar every num_processes processes, which may slow down
            # the process a bit but enable observation.

            pbar = tqdm(total=len(argument_list))
            idx = 0
            for idx in range(0, len(argument_list) // num_processes + 1):
                for result in pool.starmap(func=func, iterable=argument_list[idx*num_processes : min((idx+1)*num_processes, len(argument_list))]):
                    result_list_tqdm.append(result)
                pbar.update(min(num_processes, len(argument_list)-idx*num_processes))
                idx += 1
        else:
            for result in pool.starmap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
        pool.join()

    except:
        result_list_tqdm = list(map(func, argument_list))
    return result_list_tqdm


def save_file(f_image, save_dir, suffix='.jpg'):
    """
    Save images with designated suffix
    """
    filepath, _ = os.path.split(save_dir)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f_image.save(save_dir + suffix)


def make_path(file_pack_path):
    if not os.path.exists(file_pack_path):
        os.makedirs(file_pack_path)


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
        return res

    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


def convert_to_npy_no_opening(patch, patch_size=(960, 960)):
    """
    Convert the image into numpy format;
    The numpy size is slightly cropped
    :param patch: the patch to be converted
    :param patch_size: the required image size
    :return:
    """
    patch_size = to_2tuple(patch_size)
    img = patch
    w, h = img.size
    factor = min(w // patch_size[0], h // patch_size[1])
    numpy_img = img.crop([0, 0, factor * patch_size[0], factor * patch_size[1]])
    numpy_img = np.array(numpy_img)

    return numpy_img


class to_patch:
    """
    Split an image into patches, each patch with the size of patch_size
    """
    def __init__(self, patch_size=(16, 16)):
        patch_size = to_2tuple(patch_size)
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

    def __call__(self, x):
        x = torch.tensor(x)
        x = x.permute(2, 0, 1)
        c, h, w = x.shape
        # print(x.shape)
        # assert h // self.patch_h == h / self.patch_h and w // self.patch_w == w / self.patch_w
        num_patches = (h // self.patch_h) * (w // self.patch_w)

        h_1 = (h // self.patch_h) * self.patch_h
        w_1 = (w // self.patch_w) * self.patch_w
        x = x[:, ((h - h_1) // 2):((h - h_1) // 2 + h_1), ((w - w_1) // 2):((w - w_1) // 2 + w_1)]
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

        return patches


def to_2tuple(input):
    if type(input) is tuple:
        if len(input) == 2:
            return input
        else:
            if len(input) > 2:
                output = (input[0], input[1])
                return output
            elif len(input) == 1:
                output = (input[0], input[0])
                return output
            else:
                print('cannot handle none tuple')
    else:
        if type(input) is list:
            if len(input) == 2:
                output = (input[0], input[1])
                return output
            else:
                if len(input) > 2:
                    output = (input[0], input[1])
                    return output
                elif len(input) == 1:
                    output = (input[0], input[0])
                    return output
                else:
                    print('cannot handle none list')
        elif type(input) is int:
            output = (input, input)
            return output
        else:
            print('cannot handle ', type(input))
            raise ('cannot handle ', type(input))


def pick_patch(patch):
    """
    Pick the image patch that includes tissue information
    The patch with relatively more R is to be picked
    :param patch: input with numpy format
    :return: bool
    """
    patch = array2img(patch)
    img_single = patch.resize((1, 1), Image.BILINEAR)
    r, g, b = img_single.getpixel((0, 0))
    if r > 220 and g > 220 and b > 220:
        return False
    else:
        return True


def array2img(patch):
    img = Image.fromarray(patch.astype('uint8')).convert('RGB')
    return img


def make_name(former_name, patch_size, patch_num):
    """
    Important: each image patch's name include x, y, patch_size;
    The exact location of an image patch is (x * patch_size, y * patch_size)
    """
    former_patch_size = int(former_name.split('-')[-3])
    former_x = int(former_name.split('-')[-2])
    former_y = int(former_name.split('-')[-1])
    img_real_name = former_name[::-1].split('-', 3)[-1][::-1]

    ratio = int(former_patch_size / patch_size)
    x = patch_num % ratio if patch_num % ratio != 0 else ratio
    x = x - 1 # every coordinate starts with 0
    x = former_x * ratio + x

    y = patch_num // ratio if patch_num % ratio != 0 else patch_num // ratio - 1
    y = former_y * ratio + y

    img_name = img_real_name + '-' + str(patch_size) + '-' + str(x) + '-' + str(y)
    print(img_name)
    return img_name


def SVS_cut_to_patch(img, save_root, patch_size,
                     class_name,
                     name_dir_3840, name_dir_0, name_dir_1, name_dir_2,
                     patient_folder=False,
                     mpp=None,
                     L=True, M=True, S=True):
    slide = openslide.open_slide(img)
    img_name = os.path.split(img)[1].split('.')[0]
    if mpp is None:
        MPP = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        print(MPP, img)
    else:
        MPP = mpp
    resize_ratio = STANDARD_MPP / float(MPP)
    print(resize_ratio)
    if 1.1 > resize_ratio > 0.9:
        patch_size_num_0 = patch_size[0][0]
    else:
        patch_size_num_0 = int(patch_size[0][0] * resize_ratio)
    print(patch_size_num_0)
    save_root_0 = os.path.join(os.path.join(save_root, str(patch_size[0][0])), class_name + '-' + str(patch_size[0][0]))
    make_path(save_root_0)
    w, h = slide.level_dimensions[0]
    for i in range(1, w // patch_size_num_0 - 1):
        for j in range(1, h // patch_size_num_0 - 1):

            patch = slide.read_region((i * patch_size_num_0, j * patch_size_num_0), 0,
                                      (patch_size_num_0, patch_size_num_0))
            patch = patch.convert('RGB')
            if not 1.1 > resize_ratio > 0.9:
                patch = patch.resize(patch_size[0], Image.ANTIALIAS)
            # resize to 3840 * 3840
            img_single = patch.resize((1, 1), Image.ANTIALIAS)
            r, g, b = img_single.getpixel((0, 0))
            if r < 220 and g < 220 and b < 220 and r > 100 and b > 30 and r > g + 20:
                save_file(patch, os.path.join(save_root_0,
                                              img_name + '-' + str(patch_size[0][0]) + '-' + str(i) + '-' + str(j)))
                print(os.path.join(save_root_0, img_name + '-' + str(patch_size[0][0]) + '-' + str(i) + '-' + str(j)))
                name_dir_3840[os.path.join(save_root_0,
                                           img_name + '-' + str(patch_size[0][0]) + '-' + str(i) + '-' + str(
                                               j)) + '-' + str(resize_ratio)] = img
                if patient_folder is True:
                    save_root_patient_0 = os.path.join(save_root_0 + '-patient', img_name)
                    save_file(patch, os.path.join(save_root_patient_0,
                                                  img_name + '-' + str(patch_size[0][0]) + '-' + str(i) + '-' + str(j)))
                Image_name_XL = img_name + '-' + str(patch_size[0][0]) + '-' + str(i) + '-' + str(j)
                cut_to_patch(patch, Image_name_XL, save_root,
                             patch_size[1], patch_size[2], patch_size[3],
                             img_name, class_name,
                             name_dir_0, name_dir_1, name_dir_2,
                             patient_folder=patient_folder,
                             L=L, M=M, S=S)
            else:
                continue
    pd.DataFrame.from_dict(name_dir_3840, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root, str(patch_size[0][0])), class_name + '-' + str(patch_size[0][0]) + '.csv')
    )


def cut_to_patch(patch,
                 current_img_name,
                 save_root,
                 patch_size_0, patch_size_1, patch_size_2,
                 img_name, class_name,
                 name_dir_0, name_dir_1, name_dir_2,
                 patient_folder=True,
                 L=True, M=True, S=False
                 ):
    current_img_name = current_img_name
    numpy_img = convert_to_npy_no_opening(patch)
    patch_size_num_0 = patch_size_0[0]
    patch_size_num_1 = patch_size_1[0]
    patch_size_num_2 = patch_size_2[0]
    save_root_0 = os.path.join(os.path.join(save_root, str(patch_size_num_0)), class_name + '-' + str(patch_size_num_0))
    save_root_1 = os.path.join(os.path.join(save_root, str(patch_size_num_1)), class_name + '-' + str(patch_size_num_1))
    save_root_2 = os.path.join(os.path.join(save_root, str(patch_size_num_2)), class_name + '-' + str(patch_size_num_2))

    save_root_patient_0 = os.path.join(save_root_0 + '-patient', img_name)
    save_root_patient_1 = os.path.join(save_root_1 + '-patient', img_name)
    save_root_patient_2 = os.path.join(save_root_2 + '-patient', img_name)

    img_split_0 = to_patch(patch_size_0)
    img_patches_0 = img_split_0(numpy_img)

    img_split_1 = to_patch(patch_size_1)
    img_patches_1 = img_split_1(numpy_img)
    i = 0
    j = 0
    if L:
        # on most cases we need L-scale, which is 960 * 960
        for patch in img_patches_0:
            i = i + 1
            patch = patch.permute(1, 2, 0)
            patch = patch.numpy()
            if pick_patch(patch):
                img_name_0 = make_name(current_img_name, patch_size_num_0, i)
                save_dir_0 = os.path.join(save_root_0, img_name_0)
                print(save_dir_0)
                patch = array2img(patch)
                # patch = patch.resize((384, 384), Image.ANTIALIAS)  # 归为384*384
                # for our biggest CPIA we dont want to resize
                if patient_folder:
                    save_file(patch, os.path.join(save_root_patient_0, img_name_0))
                name_dir_0[save_dir_0] = '1'
                save_file(patch, save_dir_0)
    else:
        pass
    if M:
        # on most cases we need M-scale, which is 384 * 384
        # if M is false then S must be false
        for patch_1 in img_patches_1:
            # convert the image into numpy
            j = j + 1
            patch_1 = patch_1.permute(1, 2, 0)
            patch_1 = patch_1.numpy()
            if pick_patch(patch_1):
                # save 384*384 image
                img_name_1 = make_name(current_img_name, patch_size_num_1, j)
                save_dir_1 = os.path.join(save_root_1, img_name_1)
                print(save_dir_1)
                if S:
                    k = 0
                    img_split_2 = to_patch(patch_size_2)
                    img_patches_2 = img_split_2(patch_1)
                    for patch_2 in img_patches_2:
                        k = k + 1
                        patch_2 = patch_2.permute(1, 2, 0)
                        patch_2 = patch_2.numpy()
                        if pick_patch(patch_2):
                            if k % 10 == 0:
                                # down sampling
                                img_name_2 = make_name(img_name_1, patch_size_num_2, k)
                                patch_2 = array2img(patch_2)
                                save_dir_2 = os.path.join(save_root_2, img_name_2)
                                print(save_dir_2)
                                if patient_folder:
                                    save_file(patch_2, os.path.join(save_root_patient_2, img_name_2))
                                name_dir_2[save_dir_2] = '1'
                                save_file(patch_2, save_dir_2)
                else:
                    pass

                patch_1 = array2img(patch_1)
                if patient_folder:
                    save_file(patch_1, os.path.join(save_root_patient_1, img_name_1))
                name_dir_1[save_dir_1] = '1'
                save_file(patch_1, save_dir_1)
    else:
        pass
    pd.DataFrame.from_dict(name_dir_0, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_0)), class_name + '-' + str(patch_size_num_0) + '.csv')
    )
    pd.DataFrame.from_dict(name_dir_1, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_1)), class_name + '-' + str(patch_size_num_1) + '.csv')
    )
    pd.DataFrame.from_dict(name_dir_2, orient='index', columns=['origin path']).to_csv(
        os.path.join(os.path.join(save_root,
                                  str(patch_size_num_2)), class_name + '-' + str(patch_size_num_2) + '.csv')
    )


def read_and_convert(data_root, save_root, suffix, patient_folder, mpp, resume, resume_dataset, resume_WSI, L, M, S):

    class_names = os.listdir(data_root)
    print(class_names)

    for class_name in class_names:
        svs_class_root = os.path.join(data_root, class_name)
        svs_all_files = find_all_files(svs_class_root, suffix)

        name_dir_3840 = {}
        name_dir_0 = {}
        name_dir_1 = {}
        name_dir_2 = {}
        arg_list = []

        for img in svs_all_files:
            arg_list.append([
                img,
                save_root,
                patch_size,
                class_name,
                name_dir_3840,
                name_dir_0,
                name_dir_1,
                name_dir_2,
                patient_folder,
                mpp,
                L,
                M,
                S])
        # Run code in parallel. One process for each WSI.
        run_map_mp(SVS_cut_to_patch, arg_list, is_tqdm=True)




def get_args_parser():
    parser = argparse.ArgumentParser('CPIA_WSI', add_help=False)

    parser.add_argument('--input', default='/data/WSI_1', type=str,
                        help='path to input dataset')
    parser.add_argument('--output', default='/data-save', type=str,
                        help='path to output patches')
    parser.add_argument('--suffix', default='svs', type=str,
                        help='input image suffix')
    parser.add_argument('--patient', default=False, type=bool,
                        help='whether to generate patient folder')
    parser.add_argument('--mpp', default=None, type=str,
                        help='for some datasets the MPP need to be provided manually')
    parser.add_argument('--resume', default=False, type=bool,
                        help='resume the process')
    parser.add_argument('--resume_dataset', default='CPTAC-BRCA', type=str,
                        help='path to output patches')
    parser.add_argument('--resume_WSI', default='/data/WSI_1/CPTAC-BRCA/BRCA/20BR008-76d73924-1d5a-4d75-b776-1f1b2b.svs',
                        type=str,
                        help='the actual path of the WSI to be resumed')



    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CPIA_WSI', parents=[get_args_parser()])
    args = parser.parse_args()
    read_and_convert(args.input,
                     args.output,
                     args.suffix,
                     args.patient,
                     args.mpp,
                     args.resume, args.resume_dataset, args.resume_WSI,
                     L=True, M=True, S=True)







