import argparse
import os
import shutil
import time
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from att import *
from PIL import ImageFile
from metrics import compute_accuracy, compute_f1
from utils import str2bool, RandomErase, AverageMeter, mixup_data, mixup_loss, accuracy, confusion_matrix, plot_confusion_matrix
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--device_ids', default=[0, 1], type=str, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--is_mixup', default="False", type=str2bool, help='')
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument('--lr_type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep', 'step30'])
# parser.add_argument("--lr_decay_iters", default='30')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')

parser.add_argument('--rotate', default=True, type=str2bool)
parser.add_argument('--rotate_min', default=-180, type=int)
parser.add_argument('--rotate_max', default=180, type=int)
parser.add_argument('--rescale', default=True, type=str2bool)
parser.add_argument('--rescale_min', default=0.8889, type=float)
parser.add_argument('--rescale_max', default=1.0, type=float)
parser.add_argument('--shear', default=True, type=str2bool)
parser.add_argument('--shear_min', default=-36, type=int)
parser.add_argument('--shear_max', default=36, type=int)
parser.add_argument('--translate', default=False, type=str2bool)
parser.add_argument('--translate_min', default=0, type=float)
parser.add_argument('--translate_max', default=0, type=float)
parser.add_argument('--flip', default=True, type=str2bool)
parser.add_argument('--contrast', default=True, type=str2bool)
parser.add_argument('--contrast_min', default=0.9, type=float)
parser.add_argument('--contrast_max', default=1.1, type=float)
parser.add_argument('--random_erase', default=True, type=str2bool)
parser.add_argument('--random_erase_prob', default=0.5, type=float)
parser.add_argument('--random_erase_sl', default=0.02, type=float)
parser.add_argument('--random_erase_sh', default=0.4, type=float)
parser.add_argument('--random_erase_r', default=0.3, type=float)
best_val_ac_score = 0

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
def main():
    global args, best_val_ac_score
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    # create model
    if args.arch == "resnet":
        model = ResidualNet(args.data, args.depth, 1000, 'eclsam')
        # model = ResidualNet(args.data, args.depth, 1000, 'clsam')
        model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
        if args.data == "UCM" or args.data == "UCMdataaug":
            model.fc = torch.nn.Linear(model.fc.in_features, 21)
        elif args.data == "AID" or args.data == "AID0.5" or args.data == "AIDdataaug" or args.data == "AIDdataaug0.5":
            model.fc = torch.nn.Linear(model.fc.in_features, 30)
        elif args.data == "NWPU-RESISC45" or args.data == "NWPU-RESISC450.1" or args.data == "NWPU-RESISC45dataaug" or \
                args.data == "NWPU-RESISC45dataaug0.1":
            model.fc = torch.nn.Linear(model.fc.in_features, 45)
        # params_to_update = model.parameters()
        # params_to_update = []
        # for p in model.parameters():
        #     p.requires_grad = False
        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #         # params_to_update.append(param)
        #         print("\t", name)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    # print("model")
    # print(model)

    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # # get the number of model parameters
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))
    # print('Number of model Flops,Paras: ',flops,params)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_ac_score = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            # transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(
                degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
                translate=(args.translate_min, args.translate_max) if args.translate else None,
                scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
                shear=(args.shear_min, args.shear_max) if args.shear else None,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            # RandomErase(prob=args.random_erase_prob if args.random_erase else 0,
            #             sl=args.random_erase_sl,
            #             sh=args.random_erase_sh,
            #             r=args.random_erase_r),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # ------------_Dataload_start------------------

    if args.data == "UCM":
        train_set = datasets.ImageFolder('./data/UCM/train', transform=image_transforms['train'])
        val_set = datasets.ImageFolder('./data/UCM/val', transform=image_transforms['val'])

    elif args.data == "AID":
        train_set = datasets.ImageFolder('./data/AID/train', transform=image_transforms['train'])
        val_set = datasets.ImageFolder('./data/AID/val', transform=image_transforms['val'])

    elif args.data == "AID0.5":
        train_set = datasets.ImageFolder('./data/AID0.5/train', transform=image_transforms['train'])
        val_set = datasets.ImageFolder('./data/AID0.5/val', transform=image_transforms['val'])

    elif args.data == "NWPU-RESISC45":
        train_set = datasets.ImageFolder('./data/NWPU-RESISC45/train', transform=image_transforms['train'])
        val_set = datasets.ImageFolder('./data/NWPU-RESISC45/val', transform=image_transforms['val'])

    elif args.data == "NWPU-RESISC450.1":
        train_set = datasets.ImageFolder('./data/NWPU-RESISC450.1/train', transform=image_transforms['train'])
        val_set = datasets.ImageFolder('./data/NWPU-RESISC450.1/val', transform=image_transforms['val'])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ------------_Dataload_end--------------------

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_ac_score, train_time = train(train_loader, model, criterion, optimizer, epoch)


        # evaluate on validation set
        val_loss, val_ac_score, val_time = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_ac_score > best_val_ac_score
        best_val_ac_score = max(val_ac_score, best_val_ac_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_ac_score': best_val_ac_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix)
        print('train_time: %.4f - train_loss %.4f - train_acc %.4f - '
              'val_time: %.4f - val_loss %.4f - val_ac_score %.4f - '
              'bets_val_acc %.4f' % (train_time, train_loss, train_ac_score, val_time,
                                     val_loss, val_ac_score, best_val_ac_score))

    ratio = len(train_set) / args.batch_size / args.print_freq
    # ratio = int(ratio)+1
    ratio = int(ratio)
    show_acc_curv(ratio)

    # if args.data == "UCM":
    #     attack_type = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
    #                    'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
    #                    'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway',
    #                    'sparseresidential', 'storagetanks', 'tenniscourt']
    #     plot_confusion_matrix(classes=attack_type, cm=conf_matrix.numpy(), savename='UCM.jpg', normalize=False,
    #                           title='UCM confusion matrix')
    # elif args.data == "AID" or args.data == "AID0.5":
    #     attack_type = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial',
    #                    'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow',
    #                    'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port',
    #                    'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium',
    #                    'StorageTanks', 'Viaduct']
    #     plot_confusion_matrix(classes=attack_type, cm=conf_matrix.numpy(), savename='AID.jpg', normalize=False,
    #                           title='AID confusion matrix')
    # elif args.data == "NWPU-RESISC45" or args.data == "NWPU-RESISC450.1":
    #     attack_type = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge',
    #                    'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential',
    #                    'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor',
    #                    'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential',
    #                    'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
    #                    'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice',
    #                    'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
    #                    'terrace', 'thermal_power_station', 'wetland']
    # plot_confusion_matrix(classes=attack_type, cm=conf_matrix.numpy(), savename='NWPU.jpg', normalize=False,
    #                       title='NWPU-RESISC45 confusion matrix')


def show_acc_curv(ratio):
    # 训练准确率曲线的x、y
    train_x = list(range(len(global_train_acc)))
    print(train_x)
    train_y = global_train_acc
    # print(train_y)

    # 测试准确率曲线的x、y
    # 每ratio个训练准确率对应一个测试准确率
    test_x = train_x[ratio - 1::ratio]
    print(test_x)
    test_y = global_test_acc
    # print(test_y)

    plt.title('ResNet50 ACC')
    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')
    plt.savefig('./fig/acc_curv_' + args.prefix + '.jpg')
    # plt.show()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        inputs, targets = torch.autograd.Variable(input), torch.autograd.Variable(target)
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                    nBatch=len(train_loader), method=args.lr_type)
        output = model(inputs)

        loss = criterion(output, targets)
        pred1, pred5 = accuracy(output.data, targets, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(pred1.item(), input.size(0))
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == targets).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            traind_total = (i + 1) * len(targets)
            acc = 100. * correct / traind_total
            print('Epoch: [{0}][{1}/{2}] | '
                  'LR :{lr:.6f} | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'Prec {top1.val:.4f} ({top1.avg:.4f}) | '.format(
                epoch, i + 1, len(train_loader), lr=lr, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            global_train_acc.append(top1.avg)

    return losses.avg, top1.avg, batch_time.sum


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    ac_scores = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss

            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pred1, pred5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(pred1.item(), input.size(0))

            _, pre = torch.max(output.data, 1)
            correct += (pre == target_var).sum()

            if (i + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.4f} ({top1.avg:.4f})'.format(
                    i + 1, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
        acc = correct.item() * 100. / (len(val_loader.dataset))
        global best_acc
        best_acc = 0
        if acc > best_acc:
            best_acc = acc
        global_test_acc.append(acc)
        if best_acc > 90:
            torch.save(model.state_dict(), './state/resnet50' + args.prefix + str(best_acc) + '.pth')

    return losses.avg, top1.avg, batch_time.sum


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr, decay_rate = args.lr, 0.1
        if epoch >= args.epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= args.epochs * 0.5:
            lr *= decay_rate
    elif method == 'step30':
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    best_acc = 0
    global_train_acc = []
    global_test_acc = []
    main()