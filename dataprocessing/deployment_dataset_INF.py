"""
self supervise dataset AI-inferance    Script  ver： Aug 25th 22:00

"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.append("..")
from Backbone.getmodel import get_model
from utils.tools import find_all_files
from utils.data_augmentation import data_augmentation


def trans_csv_folder_to_imagefoder(target_path=r'C:\Users\admin\Desktop\MRAS_SEED_dataset',
                                   original_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_org_image',
                                   csv_path=r'C:\Users\admin\Desktop\dataset\MARS_SEED_Dataset\train\train_label.csv'):
    """
    Original data format: a folder with image inside + a csv file with header which has the name and category of every image.
    Process original dataset and get data packet in image folder format

    :param target_path: the path of target image folder
    :param original_path: The folder with images
    :param csv_path: A csv file with header and the name and category of each image
    """
    idx = -1
    with open(csv_path, "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for row in tqdm(rows):
            idx += 1

            item_path = row[0]
            if os.path.exists(os.path.join(target_path, row[1])):
                shutil.copy(item_path, os.path.join(target_path, row[1]))
            else:
                os.makedirs(os.path.join(target_path, row[1]))
                shutil.copy(item_path, os.path.join(target_path, row[1]))

        print('total num:', idx)


class PILImageTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        # Trans cv2 BGR image to PIL RGB image
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return Image.fromarray(np.uint8(image))


class Front_Background_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_root, data_transforms=None, edge_size=384, suffix='.jpg'):

        super().__init__()

        self.data_root = input_root

        # get files
        self.input_ids = sorted(find_all_files(self.data_root, suffix=suffix))

        # to PIL
        self.PIL_Transform = PILImageTransform()

        # get data augmentation and transform
        if data_transforms is not None:
            self.transform = data_transforms
        else:
            self.transform = transforms.Compose([transforms.Resize(edge_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # get data path
        imageName = self.input_ids[idx]
        # get image id
        imageID = imageName
        # 文件名 os.path.split(imageName)[-1].split('.')[0]

        # get data
        # CV2 0-255 hwc，in totensor step it will be transformed to chw.  ps:PIL(0-1 hwc)
        image = np.array(cv2.imread(imageName), dtype=np.float32)

        image = self.transform(self.PIL_Transform(image))

        return image, imageID


def inferance(model, dataloader, record_dir, class_names=['0', '1'], result_csv_name='inferance.csv', device='cuda'):
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    model.eval()
    print('Inferance')
    print('-' * 10)

    check_idx = 0

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        # Iterate over data.
        for images, imageIDs in dataloader:
            images = images.to(device)

            # forward
            outputs = model(images)
            confidence, preds = torch.max(outputs, 1)

            pred_labels = preds.cpu().numpy()

            for output_idx in range(len(pred_labels)):
                f_log.write(str(imageIDs[output_idx]) + ', ' + str(class_names[pred_labels[output_idx]]) + ', \n')
                check_idx += 1

        f_log.close()
        print(str(check_idx) + ' samples are all recorded')


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    # PATH
    model_idx = args.model_idx
    dataroot = args.dataroot
    save_model_path = os.path.join(args.model_path, 'CLS_' + model_idx + '.pth')
    record_dir = args.record_dir
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    gpu_idx = args.gpu_idx

    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    use_cls_token = False if args.cls_token_off else True
    use_pos_embedding = False if args.pos_embedding_off else True
    use_att_module = None if args.att_module == 'None' else args.att_module
    edge_size = args.edge_size
    batch_size = args.batch_size

    data_transforms = data_augmentation(data_augmentation_mode=args.data_augmentation_mode, edge_size=edge_size)

    inf_dataset = Front_Background_Dataset(dataroot, data_transforms=data_transforms['val'], edge_size=edge_size,
                                           suffix='.jpg')
    dataloader = torch.utils.data.DataLoader(inf_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    class_names = ['0', '1']  # 0 for empty

    # Get model
    pretrained_backbone = False
    if args.num_classes == 0:
        print("class_names:", class_names)
        num_classes = len(class_names)
    else:
        if len(class_names) == args.num_classes:
            print("class_names:", class_names)
        else:
            print('classfication number of the model mismatch the dataset requirement of:', len(class_names))
            return -1

    model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                      pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)

    # todo: this model structure is formed under only one condition
    if gpu_idx == -1:
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            except:
                print("GPU distributing ERRO occur use CPU instead")

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            except:
                print("GPU distributing ERRO occur use CPU instead")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # single card for test

    try:
        model.load_state_dict(torch.load(save_model_path), False)
    except:
        print('model loading erro')
    else:
        print('model loaded')

    model.to(device)

    inferance(model, dataloader, record_dir, class_names=class_names, result_csv_name='inferance.csv', device='cuda')


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet INF')

    # Model Name or index
    parser.add_argument('--model_idx', default='Hybrid2_384_401_testsample', type=str, help='Model Name or index')

    # MIL Stripe
    parser.add_argument('--MIL_Stripe', action='store_true', help='MIL_Stripe')

    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')

    # Abalation Studies for MSHT
    parser.add_argument('--cls_token_off', action='store_true', help='use cls_token in model structure')
    parser.add_argument('--pos_embedding_off', action='store_true', help='use pos_embedding in model structure')
    # 'SimAM', 'CBAM', 'SE' 'None'
    parser.add_argument('--att_module', default='SimAM', type=str, help='use which att_module in model structure')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default=r'/data/pancreatic-cancer-project/k5_dataset',
                        help='path to dataset')
    parser.add_argument('--model_path', default=r'/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--record_dir', default=r'/home/pancreatic-cancer-project/INF',
                        help='path to record INF csv')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    parser.add_argument('--data_augmentation_mode', default=0, type=int, help='data_augmentation_mode')

    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='use Prompt Tuning strategy instead of Finetuning')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=10, type=int, help='Prompt_Token_num')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=0, type=int, help='classification number, default 0 for auto-fit')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000

    # Test setting parameters
    parser.add_argument('--batch_size', default=1, type=int, help='testing batch_size default 1')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

    # 转换生成的csv保存到哪？
