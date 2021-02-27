import numpy as np
import torch
import random
random.seed(1991)
np.random.seed(1991)
torch.cuda.manual_seed_all(1991)
torch.manual_seed(1991)

import few_shot_model
from copy import deepcopy
from utils import BatchThreader
from ss_datalayer import SSDatalayer
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils import AverageMeter, inter_and_union, measure
from coco import COCO
from pascal import VOCSegmentationRandom
import argparse
import os
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()

# general setup
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode or evaluate mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--model', type=int, default=1, choices=[0, 2, 3, 5, 8, 9],
                    help='Test contribution, 0: baseline (B), 2: B + C1, 3: B + C2, 5: B + C1 + C2; K-shots: 8: B + C2, 9: B + C2 + C3')

# dataset setup
parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco'],
                    help='pascal or coco')
parser.add_argument('--group', type=str, default=0, choices=[0, 1, 2, 3, 'all'],
                    help='the ith in PASCAL-5i or COCO-20i')
parser.add_argument('--num_folds', type=int, default=4,
                    help='total number of folds')
parser.add_argument('--num_shots', type=int, default=1,
                    help='number of shot to test')

# network setup
parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet101', 'vgg16'],
                    help='resnet101')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--output_stride', type=int, default=8,
                    help='output stride for the ResNet output resolution compared to original image size')
parser.add_argument('--step', type=int, default=1,
                    help='step to run in contribution 2: Boosting')

# training setup
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=None,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=10000,
                    help='number of iteration to run')

# other setups
parser.add_argument('--out_dir', type=str, default='data/val',
                    help='output directory')
parser.add_argument('--val_interval', type=int, default=200,
                    help='validate interval')
args = parser.parse_args()

try:
    args.group = int(args.group)
except:
    args.group = args.group

def main():
    assert torch.cuda.is_available()
    model_fname = 'logs_data_official'
    model_fname += ('_vgg' if 'vgg' in args.backbone else '_resnet') 
    model_fname += ('_pascal' if 'pascal' in args.dataset else '_coco')
    if not os.path.isdir(model_fname):
        os.mkdir(model_fname)
        
    model_fname += '/deeplab_{0}_{1}_{5}_{3}_{4}_v3_{2}_model_{6}'.format(
        args.backbone, args.dataset, args.exp, args.group, args.num_folds, args.output_stride, args.model)

    if not os.path.isdir(model_fname):
        os.mkdir(model_fname)

    if args.dataset == 'pascal':
        dataset = VOCSegmentationRandom('datasets/VOCdevkit',
                                           train=args.train, crop_size=args.crop_size,
                                           group=args.group, num_folds=args.num_folds,
                                           batch_size=args.batch_size, num_shots=args.num_shots,
                                           iteration=args.iteration)
    elif args.dataset == 'coco':
        dataset = COCO('datasets/coco2017/',
                       train=args.train, crop_size=args.crop_size,
                       group=args.group, num_folds=args.num_folds,
                       batch_size=args.batch_size, num_shots=args.num_shots,
                       iteration=args.iteration)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if args.backbone == 'resnet101':
        model = getattr(few_shot_model, 'resnet101')(
            pretrained=(not args.scratch),
            num_groups=args.groups,
            beta=args.beta,
            os=args.output_stride,
            model=args.model)
    elif args.backbone == 'vgg16':
        model = getattr(few_shot_model, 'vgg16')(
            pretrained=(not args.scratch),
            model=args.model)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.dataset == 'pascal':
        if args.group == 'all':
            ref_imgs, query_imgs, query_labels, ref_labels, list_labels = [], [], [], [], []
            
            for i in range(args.num_folds):
                val_file = 'data/val_{}_{}_{}_new.pkl'.format(args.dataset, i, args.num_folds)
                temp_data = torch.load(val_file)
                ref_imgs.extend(temp_data[0][:250])
                ref_labels.extend(temp_data[1][:250])
                query_imgs.extend(temp_data[2][:250])
                query_labels.extend(temp_data[3][:250])
                list_labels.extend(temp_data[4][:250])
                
        else:  
            datalayer = SSDatalayer(args.group, args.num_shots)  
            val_file = 'data/val_{}_{}_{}{}_new.pkl'.format(
                args.dataset, args.group, args.num_folds, '' if args.num_shots == 1 else '_5shot')
            print(val_file)

            if args.num_shots == 1:
                if not os.path.isfile(val_file):
                    ref_imgs, query_imgs, query_labels, ref_labels, list_labels = [], [], [], [], []
                    
                    while True:

                        data = datalayer.dequeue()

                        dat = data[0]

                        semantic_label = dat['deploy_info']['second_semantic_labels'][0]
                        list_labels.append(semantic_label)

                        if args.num_shots == 1:
                            ref_img = dat['second_img'][0]
                            ref_label = dat['first_label'][0]
                            query_img = dat['first_img'][0]
                            query_label = dat['second_label'][0]

                            if query_label.sum() < 1000:
                                continue

                            ref_img, ref_label = torch.Tensor(ref_img), torch.Tensor(ref_label)
                            query_img, query_label = torch.Tensor(query_img), torch.Tensor(query_label)

                            ref_img = torch.unsqueeze(ref_img, dim=0)
                            query_img = torch.unsqueeze(query_img, dim=0)

                            ref_imgs.append(ref_img)
                            ref_labels.append(ref_label.long())
                            query_imgs.append(query_img)
                            query_labels.append(query_label.long())

                        if len(list_labels) >= 1000:
                            break

                    torch.save((ref_imgs, ref_labels, query_imgs,
                                query_labels, list_labels), val_file)

                else:
                    ref_imgs, ref_labels, query_imgs, query_labels, list_labels = torch.load(
                        val_file)
            
            else: # 5 shot:
                val_dataset = VOCSegmentationRandom('datasets/VOCdevkit', train=False, group=args.group, 
                                num_folds=args.num_folds, batch_size=args.batch_size, 
                                num_shots=args.num_shots, iteration=args.iteration,
                                crop_size=None
                                )

    elif args.dataset == 'coco':
        val_dataset = COCO('datasets/coco2017/',
                            train=False, crop_size=args.crop_size,
                            group=args.group, num_folds=args.num_folds,
                            batch_size=args.batch_size, num_shots=args.num_shots,
                            iteration=args.iteration)

    if args.train:
        writer = SummaryWriter('logs/{}'.format(model_fname))
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).cuda()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        optimizer = optim.SGD(model.module.parameters(
        ), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=args.train,
            pin_memory=True, num_workers=args.workers, drop_last=True)
        max_iter = len(dataset_loader)
        losses = AverageMeter()

        best_loss = 1e16
        best_iou = 0

        from time import time

        for i, (inputs_q, targets_q, inputs_s, targets_s, label) in enumerate(dataset_loader):
            lr = args.base_lr * (1 - float(i) / max_iter) ** 0.9
            optimizer.param_groups[0]['lr'] = lr

            inputs_q = inputs_q.cuda()
            targets_q = targets_q.cuda()
            inputs_s = inputs_s.cuda()
            targets_s = targets_s.cuda()
            label = label.cuda()

            attentions, outputs, outputs2, outputs3 = model(x=[inputs_q, targets_q, inputs_s, targets_s],
                                                            training=True, step=args.step)

            loss = criterion(outputs, targets_q)

            losses.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result_str = ('iter: {0}/{1}\t'
                            'lr: {2:.6f}\t'
                            'loss: {loss.val:.4f} ({loss.ema:.4f})\t'
                            .format(i+1, len(dataset_loader), lr, loss=losses))

            print(result_str)

            writer.add_scalar('training/loss', losses.ema, i)

            if (i + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = AverageMeter()

                    num_classes = 20 if args.dataset == 'pascal' else 80
                    tp_list = [0]*num_classes
                    fp_list = [0]*num_classes
                    fn_list = [0]*num_classes
                    iou_list = [0]*num_classes
                    
                    with torch.no_grad():
                        for k in tqdm(range(1000)):
                            if args.dataset == 'pascal':
                                ref_img, ref_label, query_img, query_label, label = ref_imgs[k], \
                                    ref_labels[k], query_imgs[k], query_labels[k], list_labels[k]
                            elif args.dataset == 'coco':
                                query_img, query_label, ref_img, ref_label, label = val_dataset[
                                    k]
                                query_img = query_img.unsqueeze(0)
                                ref_img = ref_img.unsqueeze(0)
                                query_label = query_label.unsqueeze(0)
                                ref_label = ref_label.unsqueeze(0)

                            ref_img, ref_label, query_img, query_label = ref_img.cuda(
                            ), ref_label.cuda(), query_img.cuda(), query_label.cuda()

                            attention, output, output2, _ = model(
                                x=[query_img, query_label, ref_img, ref_label], training=False, step=10)

                            # compute the loss:
                            loss = criterion(output, query_label)
                            val_losses.update(
                                loss.item(), args.batch_size)

                            output = output.argmax(1)

                            pred = output.data.cpu().numpy().astype(np.int32)
                            query_label = query_label.cpu().numpy().astype(np.int32)
                            tp, tn, fp, fn = measure(query_label, pred)

                            if args.dataset == 'pascal':
                                tp_list[label-1] += tp
                                fp_list[label-1] += fp
                                fn_list[label-1] += fn
                            else:
                                tp_list[label] += tp
                                fp_list[label] += fp
                                fn_list[label] += fn

                        iou_list = [
                            tp_list[ic] / float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1)) for ic in range(num_classes)]

                        if args.group == 'all':
                            class_indexes = list(range(20))
                        else:
                            class_indexes = list(range(args.group*5, (args.group+1)*5))
                        mIoU = np.mean(np.take(iou_list, class_indexes))
                        print('mIoU:', mIoU)

                    writer.add_scalar('val/loss', val_losses.ema, i)
                    writer.add_scalar('val/mIoU', mIoU, i)

                model.train()  # very important

                if best_loss > val_losses.ema:
                    best_loss = val_losses.ema
                    torch.save({
                        'iteration': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, model_fname + '/best_loss.pth')

                if best_iou < mIoU:
                    best_iou = mIoU
                    torch.save({
                        'iteration': i + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, model_fname + '/best_iou.pth')

                torch.save({
                    'iteration': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname + '/current.pth')

    else:
        # TODO Complete the evaluation with new dataset

        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        model.eval()

        mapping_names = [0, 1, 2, 0, 4, 2, 1, 4, 2, 2]

        checkpoint = torch.load(model_fname[:-1] + str(mapping_names[args.model]) + '/best_iou.pth')
        state_dict = {
            k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = False

        print(model_fname + '/best_iou.pth')

        num_classes = 20 if args.dataset == 'pascal' else 80
        tp_list = [0]*num_classes
        fp_list = [0]*num_classes
        fn_list = [0]*num_classes
        iou_list = [0]*num_classes

        alpha = 0.8

        with torch.no_grad():
            for k in tqdm(range(1000)):
                if args.num_shots == 1:
                    if args.dataset == 'pascal':
                        ref_img, ref_label, query_img, query_label, label = ref_imgs[k], \
                            ref_labels[k], query_imgs[k], query_labels[k], list_labels[k]

                    elif args.dataset == 'coco':
                        query_img, query_label, ref_img, ref_label, label = val_dataset[k]

                        query_img = query_img.unsqueeze(0)
                        ref_img = ref_img.unsqueeze(0)
                        query_label = query_label.unsqueeze(0)
                        ref_label = ref_label.unsqueeze(0)

                    ref_img, ref_label, query_img, query_label = ref_img.cuda(), \
                        ref_label.cuda(), query_img.cuda(), query_label.cuda()
                else:
                    query_img, query_label, ref_img, ref_label, label = val_dataset[k]

                    query_img = query_img.unsqueeze(0).cuda()
                    query_label = query_label.unsqueeze(0).cuda()
                    
                    ref_img = [x.unsqueeze(0).cuda() for x in ref_img]
                    ref_label = [x.unsqueeze(0).cuda() for x in ref_label]

                attention, output, output2, _ = model(
                    x=[query_img, query_label, ref_img, ref_label], training=False, step=10)

                if args.model in [0, 1, 2, 4, 8]:
                    output = output.argmax(1)
                else:
                    output = output2 > alpha

                pred = output.data.cpu().numpy().astype(np.int32)
                query_label = query_label.cpu().numpy().astype(np.int32)
                tp, tn, fp, fn = measure(query_label, pred)

                if args.dataset == 'pascal':
                    tp_list[label-1] += tp
                    fp_list[label-1] += fp
                    fn_list[label-1] += fn
                else:
                    tp_list[label] += tp
                    fp_list[label] += fp
                    fn_list[label] += fn

            iou_list = [
                tp_list[ic] / float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1)) for ic in range(num_classes)]

            print(iou_list)
            class_indexes = list(range(args.group*5, (args.group+1)*5))
            mIoU = np.mean(np.take(iou_list, class_indexes))
            print('mIoU:', mIoU)


if __name__ == "__main__":
    main()
