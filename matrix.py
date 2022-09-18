# _*_ coding: UTF-8 _*_
# Author: liming
 
import torch
import numpy as np
import argparse
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from utils import plot_confusion_matrix, str2bool
from PIL import Image
from att import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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

args = parser.parse_args()
# 模型权重和类别标签
weight_path = './state/resnet500.1NWPU.pth'

if args.data == "UCM":
    classes = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
            'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential',
            'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
            'storagetanks', 'tenniscourt']
elif args.data == "AID":
    classes = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial',
                       'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow',
                       'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port',
                       'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium',
                       'StorageTanks', 'Viaduct']
elif args.data == "AID0.5":
    classes = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial',
                       'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow',
                       'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port',
                       'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium',
                       'StorageTanks', 'Viaduct']
elif args.data == "NWPU-RESISC45":
    classes= ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church',
            'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course',
            'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential',
            'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland',
            'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
            'terrace', 'thermal_power_station', 'wetland']
elif args.data == "NWPU-RESISC450.1":
    classes= ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church',
                'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course',
                'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential',
                'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland',
                'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
                'terrace', 'thermal_power_station', 'wetland']
 
def LoadNet(weight_path):
    model = ResidualNet(args.data, args.depth, 1000, 'elsam')
    if args.data == "UCM" or args.data == "UCMdataaug":
        model.fc = torch.nn.Linear(model.fc.in_features, 21)
    elif args.data == "AID" or args.data == "AID0.5":
        model.fc = torch.nn.Linear(model.fc.in_features, 30)
    elif args.data == "NWPU-RESISC45" or args.data == "NWPU-RESISC450.1":
        model.fc = torch.nn.Linear(model.fc.in_features, 45)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model.cuda()
    return model

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

# 导入测试数据集
if args.data == "UCM":
    val_set = datasets.ImageFolder('./data/UCM/val', transform=image_transforms['val'])

elif args.data == "AID":
    val_set = datasets.ImageFolder('./data/AID/val', transform=image_transforms['val'])

elif args.data == "AID0.5":
    val_set = datasets.ImageFolder('./data/AID0.5/val', transform=image_transforms['val'])

elif args.data == "NWPU-RESISC45":
    val_set = datasets.ImageFolder('./data/NWPU-RESISC45/val', transform=image_transforms['val'])

elif args.data == "NWPU-RESISC450.1":
    val_set = datasets.ImageFolder('./data/NWPU-RESISC450.1/val', transform=image_transforms['val'])

val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
print('\n测试数据加载完毕\n')
 
true_label = []
pred_label = []
 
# 加载模型
model = LoadNet(weight_path)
 
for batch_idx, (image, label) in enumerate(val_loader):
    image, label = image.cuda(), label.cuda()
    output = model(image)
    pred = output.data.max(1, keepdim=True)[1]
    prediction = pred.squeeze(1)
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    print('输入图像的真实标签为:{}, 预测标签为:{}'.format(label[0]+1, prediction[0]+1))
    true_label.append(label[0]+1)
    pred_label.append(prediction[0]+1)
 
# 计算混淆矩阵并绘图
cm = confusion_matrix(true_label, pred_label)
plot_confusion_matrix(classes, cm, str(args.data) + '_confusion_matrix' + '.jpg', title='confusion matrix', normalize=True)
