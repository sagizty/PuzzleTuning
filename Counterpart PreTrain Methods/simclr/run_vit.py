import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ViTSimCLR
from simclr import SimCLR
import os

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')

    # Dataset related
    parser.add_argument('--data', metavar='DIR', default='./datasets',
                        help='path to dataset')
    parser.add_argument('--dataset-name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10', 'imagefolder', 'cpia-mini'])
    parser.add_argument('--mode', default='train',
                        help='train val test stage', choices=['train', 'val', 'test'])

    # Training related
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    # Model related
    parser.add_argument('-a', '--arch', type=str, default='vit_base_patch16_224',
                        help='model architecture.')
    parser.add_argument('--load_weight', type=str, help='model weight directory.')
    parser.add_argument('--img_size', type=int, default=224, help='image size. For vit: 224, for resnet: 96.')

    # added
    parser.add_argument('--log_dir', default=' ',
                            help='path where to tensorboard log')
    parser.add_argument('--output_dir', default=' ',
                            help='path where to store checkpoints')
    parser.add_argument('--init_weight_pth', default='', type=str,
                            help="init weight path") 
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    return parser


def main(args):
        
    if args.enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='EXVGQACCXPUIUQAE',
                       default_reciving_list=['foe3305@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)
        notify.add_text('SimCLR Training')
        notify.add_text('------')
        for a in str(args).split(','):
            notify.add_text(a)
        notify.add_text('------')
        notify.send_log()

    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("{}".format(args).replace(', ', ',\n'))

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, mode=args.mode, img_size=args.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ViTSimCLR(base_model=args.arch, out_dim=args.out_dim, load_weight=args.load_weight)

    # load weight from file
    if args.init_weight_pth:
        print(f'Loading weight from {args.init_weight_pth}...')
        init_weight = torch.load(args.init_weight_pth)
        model.load_state_dict(init_weight, strict=False)
        print('Weight loaded.')

    model = model.to(args.device)

    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)
    
    start_epoch = 0
    if args.load_weight:    # load from checkpoint
        checkpoint = torch.load(args.load_weight)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch'])
        # TODO: AAAAAAAA it become constant here!!!!!
        scheduler.last_epoch = start_epoch
        print(f"Loaded weights from: {args.load_weight}, starting epoch: {start_epoch}")

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(start_epoch, train_loader)
        # simclr.train_pretend(train_loader)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    main(args)
