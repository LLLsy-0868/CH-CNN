"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import data_loader
# import cifar100_loader
# import imagenet_loader
import loader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool
import utils
from utils import Utils, GPUTools
from LabelSmoothing import LabelSmoothingLoss
from torch.optim import lr_scheduler
import argparse
import logging
import time
import numpy as np

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--arch', type=str, default='ZSPP_GCN_assis_cifar1', help='which architecture to use')
parser.add_argument('--retrain', type=str, default=None)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

# CIFAR_CLASSES = 1000
#
# if args.set == 'imagenet':
#     CIFAR_CLASSES = 1000
# elif args.set == 'cifar10':
#     CIFAR_CLASSES = 10
# elif args.set == 'cifar100':
#     CIFAR_CLASSES = 100

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, num_group=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,groups=1 if in_planes==3 else num_group)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Basic(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, num_group=1):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride !=1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, in_planes, planes, stride=1,num_group=1):
        super(Bottleneck, self).__init__()
        width = int(planes/self.expansion)
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False,groups=1 if in_planes==3 else num_group)
        self.bn1 = nn.BatchNorm2d(width)
        if num_group == 4:
            num_group=width
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt0(nn.Module):

    def __init__(self, in_planes, planes, num_group=4, stride=1):
        super(ResNeXt0, self).__init__()
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, num_group))
        self.layer =nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt1(nn.Module):

    def __init__(self, in_planes, planes, num_group=4, stride=1):
        super(ResNeXt1, self).__init__()
        layers = []
        layers.append(Bottleneck(in_planes, planes, stride, num_group))


        self.layer =nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generated_init


    def forward(self, x):
        #generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TrainModel(object):
    def __init__(self,gpu_id):
        print('train_model')

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        np.random.seed(args.seed)

        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(args.seed)
        net = EvoCNNModel().cuda()

        print("gpu device")

        criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1)
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.complexity = 0                                             ##########
        self.file_id = os.path.basename(__file__).split('.')[0]

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self,optimizer, epoch,trainloader):
        print('train')
        self.net.train()

        lr = 0.1
        running_loss = 0.0
        total = 0
        correct = 0
        for  _, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        acc = float(correct)/total
        print('Train-Epoch:%3d,  Loss: %.3f, Acc:%d/%d = %.3f'% (epoch+1, running_loss/total, correct,total, acc))
        self.log_record('Train-Loss:%.3f, Acc:%.3f '%(running_loss/total, acc))            ######


    def test(self,validate_loader):
        self.net.eval()

        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for _, data in enumerate(validate_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()*labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()

        acc = float(correct)/total
        if acc > self.best_acc:
            self.best_acc = acc

        self.complexity = StatusUpdateTool.get_total_params(self.net)                                              #####
        print('Validate-Loss:%.3f, Acc:%d/%d = %.3f, Complexity:%d'%(test_loss/total, correct,total, acc, self.complexity))
        self.log_record('Validate-Loss:%.3f, Acc:%.3f , Complexity:%d'%(test_loss/total, acc, self.complexity))            ######
        return test_loss/total, acc


    def process(self, gpu_id,gen_no,param,_key,batch_size):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  ######
        print("begin process:",_key)
        self.log_record("_key:%s"%(_key))
        epoch = StatusUpdateTool.get_epoch_size()
        if gen_no < 3:
            total_epoch = epoch[0] #5

        elif gen_no < 6:
            total_epoch = epoch[1] #100
        elif gen_no < 10:
            total_epoch = epoch[2] #350
        else:
            total_epoch = epoch[3]
        lr = 0.1

        T_0 = total_epoch


        file = './retrain/%s/checkpoint.pth' % (_key)
        if os.path.exists(file):
            print("try load model param")
            load_save_point_dict = torch.load(file,map_location=torch.device('cpu'))
            temp_epoch = load_save_point_dict['epoch']
            if temp_epoch < total_epoch:
                if temp_epoch == 0: lr = 0.01
                if temp_epoch > 0: lr = 0.1;
                if temp_epoch > 148: lr = 0.01
                if temp_epoch > 248: lr = 0.001
                args.retrain = './retrain/%s' % (_key)

        if args.retrain is not None:
            print("retrain")
            load_save_point_dict = torch.load(file,map_location=torch.device('cpu'))
            self.net.load_state_dict(load_save_point_dict['state_dict'])
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            temp_epoch = load_save_point_dict['epoch']
            best_acc = load_save_point_dict['best_acc']
            optimizer.load_state_dict(load_save_point_dict['optimizer'])

            logging.info('restart from epoch: {}'.format(temp_epoch))
        else:
            best_acc = 0.0
            temp_epoch = 0

        # trainloader, validate_loader = data_loader.get_train_valid_loader('/home/siyi/CNN0.4/cifar-10-python',
        #                                                                   batch_size=batch_size, augment=True, valid_size=0.1,
        #                                                                   shuffle=True, random_seed=2312390,
        #                                                                   show_sample=False, num_workers=0,
        #                                                                   pin_memory=True)
        trainloader, validate_loader = loader.get_train_valid_loader(batch_size=batch_size, augment=True, valid_size=0.1,
                                                                     shuffle=True, random_seed=2312390,
                                                                     show_sample=False, num_workers=0, pin_memory=True)


        for p in range(temp_epoch,total_epoch):
            if temp_epoch == 0:
                lr = 0.01
            if temp_epoch > 0:
                lr = 0.1
            if temp_epoch > 148: lr = 0.01
            if temp_epoch > 248: lr = 0.001
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            self.train(optimizer, p,trainloader)

            val_loss,val_acc = self.test(validate_loader)
            is_best = False
            if val_acc > best_acc:
                best_acc = val_acc
                is_best = True
            model_dir = './retrain/%s/checkpoint.pth' % (_key)
            print("model_dir",model_dir)
            dir ='./retrain/%s' % (_key)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save({
                    'epoch': p + 1,
                    'state_dict': self.net.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
            }, model_dir)
            # scheduler.step()
                                                                                                                ######
        print('\t best_acc:%f, complexity:%d'%(self.best_acc , self.complexity))                                     ######

        return self.best_acc , self.complexity                                                                       ######



class RunModel(object):
    def do_work(self, gpu_id, file_id, _key, _str, gen_no,param):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        complexity = 0

        epoch = StatusUpdateTool.get_epoch_size()
        if gen_no < 5:
            total_epoch = epoch[0]
        elif gen_no < 8:
            total_epoch = epoch[1]  # 100
        elif gen_no < 10:
            total_epoch = epoch[2]  # 350
        else:
            total_epoch = epoch[3]

        temp_epoch = 0
        file = './retrain/%s/checkpoint.pth' % (_key)
        if os.path.exists(file):
            load_save_point_dict = torch.load(file,map_location=torch.device('cpu'))
            temp_epoch = load_save_point_dict['epoch']

        batch_size = 512
        times = 0
        while temp_epoch < total_epoch:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            try:
                print("run_model")
                m = TrainModel(gpu_id)
                m.log_record('worker name:%s[%d]'%( multiprocessing.current_process().name, os.getpid()), first_time=True)

                print("temp_epoch",temp_epoch, file_id)

                # print("try m.process")
                best_acc, complexity = m.process(gpu_id,gen_no,param,_key,batch_size)
                file = './retrain/%s/checkpoint.pth' % (_key)
                if os.path.exists(file):
                    load_save_point_dict = torch.load(file, map_location=torch.device('cpu'))
                    temp_epoch = load_save_point_dict['epoch']
                    args.retrain = './retrain/%s' % (_key)

                if temp_epoch == total_epoch:
                    Utils.save_indi_to_cache(_key, _str, best_acc, complexity)
                    break


            except BaseException as exception:
                print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), repr(exception)))
                m.log_record('Exception occur:%s'%(repr(exception)))
                e = "out of memory"
                if e in repr(exception):
                    if batch_size == 1:
                        break
                    batch_size = int(3 * batch_size / 4)
                    print("batch_size is 3/4: ", batch_size)
                else:
                    break

            finally:
                print("read the file")


        m.log_record('Finished-Acc:%.3f,Complexity:%d' % (best_acc, complexity))
        Utils.save_indi_to_cache(_key, _str, best_acc, complexity)
        key = file_id
        if os.path.exists('./populations/after_%s.txt'%(file_id[4:6])):
            print("file exist true")
            f = open('./populations/after_%s.txt'%(file_id[4:6]), "r")
            lines = f.readlines()
            indi_exist = False
            for line in lines:
                if key == line[0:8]:
                    indi_exist = True
            if indi_exist== False:
                f_a = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
                f_a.write('%s,%.5f,%d\n'%(file_id, best_acc, complexity))
                f_a.flush()
                f_a.close()
            else:
                f_w = open('./populations/after_%s.txt'%(file_id[4:6]), "w")
                for line in lines:
                    if key == line[0:8]:
                        print("train",line[0:8])
                        str = '%s,%.5f,%d\n'%(file_id, best_acc, complexity)
                        line = line.replace(line, str)
                    f_w.write(line)
                f_w.flush()
                f_w.close()
            f.flush()
            f.close()
        else:
            f = open('./populations/after_%s.txt'%(file_id[4:6]), "a+")
            f.write('%s,%.5f,%d\n'%(file_id, best_acc, complexity))
            f.flush()
            f.close()


"""


