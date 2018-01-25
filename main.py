'''
@file: run.py
@version: v1.0
@date: 2018-01-07
@author: ruanxiaoyi
@brief: Run the network
@remark: {when} {email} {do what}
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from common.net import VGGNet
from common.resnet import resnet18
from common.resnet import resnet50
from common.resnet import resnet34
from common.resnet import resnet101
from common.resnet import resnet152
from common.resnet import ResNet
from common.densenet import DenseNet
from common.dataset import TIN200Data
from common.utils import localtime, save
import torchvision.models as models
import torch.backends.cudnn as cudnn


def train(model, loss_fn, optimizer, lr_schedule, num_epochs=1, loader=None, val_loader=None):
    num_correct = 0
    num_samples = 0
    best_val_acc = 0
    acc = 0
    val_acc = 0
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader):
            x_train = Variable(x.cuda())
            y_train = Variable(y.cuda())

            scores = model(x_train)
            loss = loss_fn(scores, y_train)

            # reference https://discuss.pytorch.org/t/argmax-with-pytorch/1528
            _, preds = scores.data.cpu().max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            if (t + 1) % 20 == 0:
                print('t = %d, loss = %.4f, acc = %.4f%%' %
                      (t + 1, loss.data[0], 100 * acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = check_accuracy(model, val_loader)
        #lr_schedule.step(val_acc, epoch=epoch + 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("saving net.....")
            save(model, True, True)
        adjust_learning_rate(optimizer,epoch)
        print('-------------------------------')
        print("The best validation accuracy:%.4f%%" % (100 * best_val_acc))
        print('-------------------------------')


def check_accuracy(model, loader):
    print('Checking accuracy on validation set')

    val_correct = 0
    val_samples = 0
    # Put the model in test mode (the opposite of model.train(), essentially)
    model.eval()
    for x, y in loader:
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_var = Variable(x, volatile=True)

        scores = model(x_var.type(torch.cuda.FloatTensor))
        _, preds = scores.data.cpu().max(1)
        val_correct += (preds == y).sum()
        val_samples += preds.size(0)
    val_acc = float(val_correct) / val_samples
    print('Got %d / %d correct (%.4f%%)' %
          (val_correct, val_samples, 100 * val_acc))
    return val_acc


def predict(model, loader):
    from os.path import join as pjoin

    print('Predicting on test set')
    classid = []
    test_img_name = []
    classid_map = {}

    for _, _, files in os.walk(pjoin('/data1/tiny-imagenet-200', 'test')):
        if files:
            test_img_name = files

    with open(pjoin('/data1/tiny-imagenet-200', 'wnids.txt'), 'r') as f:
        content = [x.strip() for x in f.readlines()]
        classid_map = {index: classid for index,
                       classid in enumerate(content)}

    model.eval()
    for x in loader:
        x_var = Variable(x, volatile=True)
        scores = model(x_var.type(torch.cuda.FloatTensor))
        _, preds = scores.data.cpu().max(1)
        classid += [classid_map[p] for p in preds]

    with open(pjoin(os.getcwd(), 'predictions', localtime(), '.txt'), 'w') as f:
        for i in len(classid):
            f.write(f'{test_img_name[i]} {classid[i]}\n')


def adjust_learning_rate(optimizer, num_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.9 ** (num_epoch // 5))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    use_cuda = torch.cuda.is_available()

    train_datasets = TIN200Data('/home/zgj/Tiny/')
    val_datasets = TIN200Data('/home/zgj/Tiny/', 'val')

    # test_datasets = TIN200Data(
    #     './tiny-imagenet-200', './tiny-imagenet-200/wnids.txt', 'test')

    train_loader = data.DataLoader(
        train_datasets, batch_size=1000, shuffle=True, num_workers=5)
    val_loader = data.DataLoader(
        val_datasets, batch_size=1000, shuffle=True, num_workers=5)

    #net = VGGNet()
    #net = models.resnet34()
    #net.conv1 = nn.Conv2d(3,64,kernel_size = 3,stride=1, padding=1 ,bias=False)
    #net.fc = nn.Linear(4096,200)
    #net = DenseNet(64, 28, 0.4, 200, 64)
    #net = resnet18()
    #net = resnet50()    
    #net = resnet18(num_classes=200)
    #net = resnet152()
    #net.cuda()
    #net = ResNet(Bottleneck,[3,4,23,3],num_classes=200)
    
    resnet = models.resnet101(pretrained=True)
    net = resnet101()

    #提取fc层中固定的参数
    fc_features = resnet.fc.in_features
    #修改类别为9
    resnet.fc = nn.Linear(fc_features, 200)

    #读取参数
    pretrained_dict = resnet.state_dict()
    model_dict = net.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载我们真正需要的state_dict
    net.load_state_dict(model_dict)

    #fc_features = net.fc.in_features
    #修改类别为200
    #net.fc = nn.Linear(fc_features, 200)
    print(net)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    lr = 0.02
    optimizer = optim.SGD(params=net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    #optimizer = optim.Adam(params=net.parameters(), lr=0.1, weight_decay = 5e-3)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True, patience=5)

    loss_fn = nn.CrossEntropyLoss()

    train(net, loss_fn, optimizer,  lr_schedule, num_epochs=100,
          loader=train_loader, val_loader=val_loader)
    #check_accuracy(net, val_loader)

    #save(net)


if __name__ == '__main__':
    main()
