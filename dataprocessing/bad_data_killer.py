"""
datacheck via dataloader     Script  ver： Feb 23th 21:00
loop the data and check if they are all cool
"""
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn.functional as func
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision import models
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context


def data_loop(device, train_loader, check_minibatch=100):
    model_time = time.time()
    prev_time = model_time
    index = 0

    for data, label in train_loader:
        data = data.to(device)

        # at the checking time now
        if index % check_minibatch == check_minibatch - 1:
            check_index = index // check_minibatch + 1
            now_time = time.time()
            gap_time = now_time - prev_time
            prev_time = now_time
            print('index of ' + str(check_minibatch) + ' minibatch:', check_index, '     time used:', gap_time)

        index += 1

    print('all checked, time used:', time.time() - model_time)


if __name__ == '__main__':
    data_path = r'/root/autodl-tmp/datasets/L'
    edge_size = 224
    transform_train = transforms.Compose([transforms.Resize([edge_size, edge_size]),transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_path, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=500, shuffle=False, num_workers=32)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loop(device, train_loader)
