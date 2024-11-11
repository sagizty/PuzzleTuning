import argparse
import torch
import os
import numpy as np
import random
import torch.nn as nn
from util.Dataset import FoldDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model.JigsawNet import JigsawNetViT
from tqdm import tqdm

import util.misc as misc

import builtins

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def save_checkpoint(net, ckpt_pth, epoch):
    try:
        os.makedirs(ckpt_pth)
        print(f'Created checkpoint directory: {ckpt_pth}')
    except OSError:
        pass

    checkpoint_name = f'checkpoint_{epoch}.pth'
    torch.save(net.state_dict(), os.path.join(ckpt_pth, checkpoint_name))

    print(f'Checkpoint {checkpoint_name} saved !')

def load_checkpoint(net, ckpt_pth):

    if not os.path.exists(ckpt_pth):
        return net, 1

    # Search for files in the current directory
    # Filter files that match the pattern "checkpoint_{epoch}"
    checkpoint_files = [file for file in os.listdir(ckpt_pth) if file.startswith('checkpoint_')]

    # Extracting the epoch numbers and finding the maximum
    max_epoch = 1
    max_epoch_file = None
    for file in checkpoint_files:
        # Extract epoch number from the file name
        try:
            epoch_num = int(file.split('_')[1].split('.')[0])
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                max_epoch_file = os.path.join(ckpt_pth, file)
        except ValueError:
            # In case the file name does not have a valid integer after 'checkpoint_'
            continue
    if not max_epoch_file:
        print(f'no checkpoint founded, skip...')

    else:
        checkpoint = torch.load(max_epoch_file, map_location='cpu')
        net.load_state_dict(checkpoint)
        print(f'Checkpoint {max_epoch_file} loaded !')

    return net, max_epoch

def evaluate(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        all = 0
        p = 0
        for batch in test_loader:
            clips, labels = batch
            clips = clips.to(device)

            # ---- forward ----
            pred = model(clips) # B * 1000

            pred_label = torch.argmax(torch.softmax(pred.detach().cpu(), dim=1), dim=1).long()

            p += (pred_label == labels).sum().item()
            all += labels.size(0)

    return p/all


def train(train_loader, test_loader, model, optimizer, epochs, device, writer, ckpt_pth='./log', start_epoch=1, global_rank=0):
    # ----prepare ----
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # ---- training ----
    for epoch in range(start_epoch, epochs+1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        loss_stack = []
        current_step = 0
        accum_step = 0

        pbar = tqdm(total=len(train_loader), desc=f'epoch[{epoch}/{epochs}]:') if global_rank == 0 else None

        for batch in train_loader:

            # ---- data prepare ----
            clips, labels = batch
            clips = clips.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # ---- forward ----
            with torch.cuda.amp.autocast():
                preds = model(clips) # B * 1000

                # ---- loss ----
                loss = criterion(preds, labels)

            # ---- backward ----
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ---- log ----

            current_step += 1
            accum_step += 1
            if global_rank == 0 and (current_step % 20 == 0 or current_step == len(train_loader)):
                # print(f'epoch[{epoch}/{epochs}] batch[{current_step}/{len(train_loader)}] | loss (batch): {loss.item()}')
                loss_stack.append(loss.detach().cpu())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(min(20, accum_step))
                accum_step = 0

        if pbar: 
            pbar.close()

        # ---- validation ----
        test_loader.sampler.set_epoch(epoch)
        accuracy = evaluate(model, test_loader, device)

        if global_rank == 0:
            loss_mean = torch.mean(torch.stack(loss_stack), dim=0)
            writer.add_scalar('info/loss', loss_mean, epoch)
            writer.add_scalar('eval/ac', accuracy, epoch)
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch)
            print(f"""epoch[{epoch}/{epochs+1}] | performance {accuracy} | mean loss: {loss_mean}""")

        if epoch > 0 and epoch % 10 == 0 and global_rank == 0:
            save_checkpoint(model, ckpt_pth, epoch)

    print('training finish')
    return accuracy


def main(args):
    # ---- init ----

    misc.init_distributed_mode(args)
    device = torch.device('cuda')
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    eff_batch_size = args.batch_size * misc.get_world_size()
    print("effective batch size: %d" % eff_batch_size)

    imgs_dir = args.data_path
    log_path = args.log_path
    ckpt_pth = os.path.join(log_path, args.exp_name, 'checkpoints')

    try:
        os.makedirs(log_path)
    except:
        pass

    # ---- random seed ----
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ---- log & dataset ----
    if global_rank == 0:
        if not os.path.exists(os.path.join(log_path, args.exp_name)):
            os.makedirs(os.path.join(log_path, args.exp_name))

        # if os.path.exists(os.path.join(log_path, args.exp_name)):
        #     shutil.rmtree(os.path.join(log_path, args.exp_name))
        writer = SummaryWriter(os.path.join(log_path, args.exp_name))
        setup_notify()
    else:
        writer = None

    permutations = np.load('permutations.npy').tolist()

    train_pool = []
    test_pool = []
    for dirpath, dirnames, filenames in os.walk(imgs_dir):
        # Filter out jpeg files and add to the list
        for filename in [f for f in filenames if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]:
            train_pool.append(os.path.join(dirpath, filename))
            if len(test_pool) < 1000:
                test_pool.append(os.path.join(dirpath, filename))

    train_set = FoldDataset(train_pool, permutations, in_channels=3)
    test_set = FoldDataset(test_pool, permutations, in_channels=3)
    train_datasampler = DistributedSampler(train_set, rank=global_rank, shuffle=True)
    test_datasampler = DistributedSampler(test_set, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=train_datasampler)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=test_datasampler)

    # ---- model ----
    model = JigsawNetViT(3, 1000, backbone='vit', weight_file=args.init_weight)
    model = model.to(device)
    model = DDP(model, device_ids=[device], output_device=global_rank)

    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    epochs = args.epochs

    if args.ckpt:
        model, start_epoch = load_checkpoint(model, ckpt_pth)
    else:
        start_epoch = 1

    # train
    print(f'''
            training start! 
            train set num: {len(train_set)} 
            val set num: {len(test_set)}
            
            ''')

    ac = train(train_loader, test_loader, model, optimizer, epochs, device, writer, ckpt_pth, start_epoch, global_rank)

    if global_rank == 0:
        save_checkpoint(model, ckpt_pth, 'final')


def get_args_parser():

    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data_path', type=str, default='/home/lyusq/data/CPIA_mini/TEST', help='Dataset path')
    parser.add_argument('--log_path', type=str, default='./log', help='Log path')
    parser.add_argument('--init_weight', type=str, default=None, help='Init model weight')

    # training config
    parser.add_argument('-b', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('-n', '--exp_name', type=str, default='exp', dest='exp_name')
    parser.add_argument('-e', '--epochs', type=int, default=100, dest='epochs')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='seed')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # added
    parser.add_argument('--ckpt', action='store_true', help='whether to load checkpoint')

    return parser

def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_notify():
    import notifyemail as notify

    notify.setup(mail_host='smtp.163.com', 
             mail_user='xxx@163.com', 
             mail_pass='xxx',
             log_root_path='log', 
             mail_list=['xxx@163.com'])

    notify.add_text('Jigsaw Training')

if __name__ == '__main__':
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    setup_seed(42)
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
    