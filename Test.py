"""
Testing  Script  ver： Oct 23rd 17:30
"""

from __future__ import print_function, division

import argparse
import json
import time

import torchvision
from tensorboardX import SummaryWriter

from Backbone.getmodel import get_model
from Backbone.GetPromptModel import build_promptmodel

from utils.data_augmentation import *
from utils.visual_usage import *


def test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_idx, test_model_idx, edge_size,
               check_minibatch=100, device=None, draw_path='../imaging_results', enable_attention_check=None,
               enable_visualize_check=True, writer=None):
    """
    Testing iteration

    :param model: model object
    :param test_dataloader: the test_dataloader obj
    :param criterion: loss func obj
    :param class_names: The name of classes for priting
    :param test_dataset_size: size of datasets

    :param model_idx: model idx for the getting trained model
    :param edge_size: image size for the input image
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.

    :param device: cpu/gpu object
    :param draw_path: path folder for output pic
    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics

    :param writer: attach the records to the tensorboard backend
    """

    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    print('Epoch: Test')
    print('-' * 10)

    phase = 'test'
    index = 0
    model_time = time.time()

    # initiate the empty json dict
    json_log = {'test': {}}

    # initiate the empty log dict
    log_dict = {}
    for cls_idx in range(len(class_names)):
        log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

    model.eval()  # Set model to evaluate mode

    # criterias, initially empty
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in test_dataloader:  # use different dataloder in different phase
        inputs = inputs.to(device)
        # print('inputs[0]',type(inputs[0]))

        labels = labels.to(device)

        # zero the parameter gradients only need in training
        # optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # log criterias: update
        log_running_loss += loss.item()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Compute recision and recall for each class.
        for cls_idx in range(len(class_names)):
            # NOTICE remember to put tensor back to cpu
            tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                        (preds == cls_idx).cpu().numpy().astype(int))
            tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                        (preds != cls_idx).cpu().numpy().astype(int))

            fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

            fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

            # log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            log_dict[class_names[cls_idx]]['tp'] += tp
            log_dict[class_names[cls_idx]]['tn'] += tn
            log_dict[class_names[cls_idx]]['fp'] += fp
            log_dict[class_names[cls_idx]]['fn'] += fn

        # attach the records to the tensorboard backend
        if writer is not None:
            # ...log the running loss
            writer.add_scalar(phase + ' minibatch loss',
                              float(loss.item()),
                              index)
            writer.add_scalar(phase + ' minibatch ACC',
                              float(torch.sum(preds == labels.data) / inputs.size(0)),
                              index)

        # at the checking time now
        if index % check_minibatch == check_minibatch - 1:
            model_time = time.time() - model_time

            check_index = index // check_minibatch + 1

            epoch_idx = 'test'
            print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                  check_index, '     time used:', model_time)

            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            # how many image u want to check, should SMALLER THAN the batchsize

            if enable_attention_check:
                try:
                    check_SAA(inputs, labels, model, model_idx, edge_size, class_names, num_images=1,
                              pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                              draw_path=draw_path, writer=writer)
                except:
                    print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
            else:
                pass

            if enable_visualize_check:
                visualize_check(inputs, labels, model, class_names, num_images=-1,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1
    # json log: update
    json_log['test'][phase] = log_dict

    # log criterias: print
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    for cls_idx in range(len(class_names)):
        # calculating the confusion matrix
        tp = log_dict[class_names[cls_idx]]['tp']
        tn = log_dict[class_names[cls_idx]]['tn']
        fp = log_dict[class_names[cls_idx]]['fp']
        fn = log_dict[class_names[cls_idx]]['fn']
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        fp_plus_tn = fp + tn
        fn_plus_tn = fn + tn

        # precision
        if tp_plus_fp == 0:
            precision = 0
        else:
            precision = float(tp) / tp_plus_fp * 100
        # recall
        if tp_plus_fn == 0:
            recall = 0
        else:
            recall = float(tp) / tp_plus_fn * 100

        # TPR (sensitivity)
        TPR = recall

        # TNR (specificity)
        # FPR
        if fp_plus_tn == 0:
            TNR = 0
            FPR = 0
        else:
            TNR = tn / fp_plus_tn * 100
            FPR = fp / fp_plus_tn * 100

        # NPV
        if fn_plus_tn == 0:
            NPV = 0
        else:
            NPV = tn / fn_plus_tn * 100

        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
        print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
        print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
        print('{} TP: {}'.format(class_names[cls_idx], tp))
        print('{} TN: {}'.format(class_names[cls_idx], tn))
        print('{} FP: {}'.format(class_names[cls_idx], fp))
        print('{} FN: {}'.format(class_names[cls_idx], fn))

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, test_model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)

    return model


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multiple GPU

    enable_notify = args.enable_notify  # False
    enable_tensorboard = args.enable_tensorboard  # False

    enable_attention_check = args.enable_attention_check  # False
    enable_visualize_check = args.enable_visualize_check  # False

    data_augmentation_mode = args.data_augmentation_mode  # 0

    # Prompt
    PromptTuning = args.PromptTuning  # None  "Deep" / "Shallow"
    Prompt_Token_num = args.Prompt_Token_num  # 20
    PromptUnFreeze = args.PromptUnFreeze  # False

    model_idx = args.model_idx  # the model we are going to use. by the format of Model_size_other_info

    # structural parameter
    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    use_cls_token = False if args.cls_token_off else True
    use_pos_embedding = False if args.pos_embedding_off else True
    use_att_module = None if args.att_module == 'None' else args.att_module

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot
    model_path_by_hand = args.model_path_by_hand  # None
    # Pre_Trained model basic for prompt turned model's test
    Pre_Trained_model_path = args.Pre_Trained_model_path  # None

    # CLS_ is for the CLS trained models, MIL_Stripe will be MIL trained and use Stripe to test
    test_model_idx = 'CLS_' + model_idx + '_test'
    # NOTICE: MIL model should only be tested in stripe model in this test.py

    draw_path = os.path.join(draw_root, test_model_idx)

    # load Finetuning trained model by its task-based saving name,
    # also support MIL-SI model but the MIL_Stripe is required
    if model_path_by_hand is None:
        # CLS_ is for the CLS training, MIL will be MIL training
        save_model_path = os.path.join(model_path, 'CLS_' + model_idx + '.pth')
    else:
        save_model_path = model_path_by_hand

    if not os.path.exists(draw_path):
        os.makedirs(draw_path)

    # choose the test dataset
    test_dataroot = os.path.join(dataroot, 'test')

    # dataset info
    num_classes = args.num_classes  # default 0 for auto-fit
    edge_size = args.edge_size

    # validating setting
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()

    # Data Augmentation is not used in validating or testing
    data_transforms = data_augmentation(data_augmentation_mode, edge_size=edge_size)

    # test setting is the same as the validate dataset's setting
    test_datasets = torchvision.datasets.ImageFolder(test_dataroot, data_transforms['val'])
    test_dataset_size = len(test_datasets)
    # skip minibatch none to draw 20 figs
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else test_dataset_size // (
            20 * batch_size)

    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    class_names = [d.name for d in os.scandir(test_dataroot) if d.is_dir()]
    class_names.sort()

    if num_classes == 0:
        print("class_names:", class_names)
        num_classes = len(class_names)
    else:
        if len(class_names) == num_classes:
            print("class_names:", class_names)
        else:
            print('classfication number of the model mismatch the dataset requirement of:', len(class_names))
            return -1

    # get model
    pretrained_backbone = False  # model is trained already, pretrained backbone weight is useless here

    if PromptTuning is None:
        model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                          pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)
    else:
        if Pre_Trained_model_path is not None and os.path.exists(Pre_Trained_model_path):
            base_state_dict = torch.load(Pre_Trained_model_path)
        else:
            base_state_dict = 'timm'
            print('base_state_dict of timm')

        print('Test the PromptTuning of ', model_idx)
        print('Prompt VPT type:', PromptTuning)
        model = build_promptmodel(num_classes, edge_size, model_idx, Prompt_Token_num=Prompt_Token_num,
                                  VPT_type=PromptTuning, base_state_dict=base_state_dict)

    try:
        if PromptTuning is None:
            model.load_state_dict(torch.load(save_model_path))
        else:
            if PromptUnFreeze:
                model.load_state_dict(torch.load(save_model_path))
            else:
                model.load_prompt(torch.load(save_model_path))

        print("model loaded")
        print("model :", model_idx)

    except:
        try:
            model = nn.DataParallel(model)

            if PromptTuning is None:
                model.load_state_dict(torch.load(save_model_path))
            else:
                if PromptUnFreeze:
                    model.load_state_dict(torch.load(save_model_path))
                else:
                    model.load_prompt(torch.load(save_model_path))

            print("DataParallel model loaded")
        except:
            print("model loading erro!!")
            return -1

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

    model.to(device)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None

    # if you want to run tensorboard locally
    # nohup tensorboard --logdir=/home/experiments/runs --host=0.0.0.0 --port=7777 &

    print("*********************************{}*************************************".format('setting'))
    print(args)

    test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_idx=model_idx,
               test_model_idx=test_model_idx, edge_size=edge_size, check_minibatch=check_minibatch,
               device=device, draw_path=draw_path, enable_attention_check=enable_attention_check,
               enable_visualize_check=enable_visualize_check, writer=writer)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='ViT_base', type=str, help='Model Name or index')

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
    parser.add_argument('--dataroot', default=r'/data/k5_dataset',
                        help='path to dataset')
    parser.add_argument('--model_path', default=r'/home/saved_models',
                        help='root path to save model state-dict, model will be find by name')
    parser.add_argument('--draw_root', default=r'/home/runs',
                        help='path to draw and save tensorboard output')
    # model_path_by_hand
    parser.add_argument('--model_path_by_hand', default=None, type=str, help='specified path to a model state-dict')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')

    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    parser.add_argument('--data_augmentation_mode', default=0, type=int, help='data_augmentation_mode')

    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='use Prompt Tuning strategy instead of Finetuning')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num')
    # PromptUnFreeze
    parser.add_argument('--PromptUnFreeze', action='store_true', help='prompt tuning with all parameaters un-freezed')
    # prompt model basic model path
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=0, type=int, help='classification number, default 0 for auto-fit')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000

    # Test setting parameters
    parser.add_argument('--batch_size', default=1, type=int, help='testing batch_size default 1')
    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
