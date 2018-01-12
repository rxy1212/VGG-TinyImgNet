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
from common.senet import SENet18
from common.densenet import DenseNet
from common.dataset import TIN200Data
from common.utils import localtime, save
import torchvision.models as models
import torch.backends.cudnn as cudnn


def train(model, loss_fn, optimizer, num_epochs=1, epochs = 0, loader=None):
    num_correct = 0
    num_samples = 0
    #for epoch in range(num_epochs):
    print('Starting epoch %d / %d' % (epochs + 1, num_epochs))
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
            print('t = %d, loss = %.4f, acc = %.4f' %
                    (t + 1, loss.data[0], acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(model, loader):
    print('Checking accuracy on validation set')

    num_correct = 0
    num_samples = 0
    # Put the model in test mode (the opposite of model.train(), essentially)
    model.eval()
    for x, y in loader:
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_var = Variable(x, volatile=True)

        scores = model(x_var.type(torch.cuda.FloatTensor))
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    use_cuda = torch.cuda.is_available()

    train_datasets = TIN200Data(
        '/data1/tiny-imagenet-200', '/data1/tiny-imagenet-200/wnids.txt')
    val_datasets = TIN200Data('/data1/tiny-imagenet-200',
                              '/data1/tiny-imagenet-200/wnids.txt', 'val')
    # test_datasets = TIN200Data(
    #     './tiny-imagenet-200', './tiny-imagenet-200/wnids.txt', 'test')

    train_loader = data.DataLoader(
        train_datasets, batch_size=64, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(
        val_datasets, batch_size=120, shuffle=True, num_workers=4)

    #net = VGGNet()
    #net = models.resnet18()
    #net.conv1 = nn.Conv2d(3,64,kernel_size = 3,stride=1, padding=1 ,bias=False)
    #net.fc = nn.Linear(4096,200)
    net = VGGNet()
    #net = SENet18()
    #net.cuda()
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    #optimizer = optim.SGD(params=net.parameters(), lr=7e-3, momentum=0.99, weight_decay= 5e-5, nesterov=True)
    #optimizer = optim.Adam(params=net.parameters(), lr=5e-3, weight_decay=5e-3)
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.99, weight_decay=0.01)

    loss_fn = nn.CrossEntropyLoss()
    num_epochs =30
    for epoch in range(num_epochs):
        train(net, loss_fn, optimizer, num_epochs=num_epochs, epochs=epoch, loader=train_loader)
        check_accuracy(net, val_loader)

    save(net)


if __name__ == '__main__':
    main()