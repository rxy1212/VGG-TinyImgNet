import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common.dataset import TIN200Data
# from common.net import Vgg19
# from common.net import Vgg11
from common.net import GoogleNet

gpu_type = torch.cuda.FloatTensor

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=64, eps=1e-06, momentum=0.9),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=128, eps=1e-06, momentum=0.9),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=256, eps=1e-06, momentum=0.9),
                        # nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=512, eps=1e-06, momentum=0.9),
                        # nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv5 = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(num_features=512, eps=1e-06, momentum=0.9),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.fc = nn.Sequential(
                        Flatten(), 
                        nn.Linear(8*8*512, 4096),
                        nn.ReLU(),
                        nn.Dropout2d(p=0.5, inplace=True), 
                        nn.Linear(4096, 1024), 
                        nn.ReLU(),
                        nn.Dropout2d(p=0.5, inplace=True),
                        nn.Linear(1024, 1024), 
                        nn.Linear(1024, 512),
                        nn.Linear(512, 200),
                        nn.Softmax(), 
                        )               

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x

class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout2d(p=0.25),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout2d(p=0.25),
                        Flatten(),
                        nn.Linear(16*16*64, 4096),
                        nn.Linear(4096, 2048),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Dropout2d(p=0.5),
                        nn.Linear(1024, 512),
                        nn.Linear(512, 512),
                        nn.Linear(512, 200),
                        nn.Softmax(),
                    )
    def forward(self, x):
        x = self.conv(x)
        return x


def train(model, loss_fn, optimizer, loader=None):
    num_correct = 0
    num_samples = 0
    model.train()
    for t, (x, y) in enumerate(loader):
        x_train = Variable(x.cuda())
        y_train = Variable(y.cuda())

        scores = model(x_train)
        loss = loss_fn(scores, y_train)

        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if (t + 1) % 200 == 0:
            print('t = %d, loss = %.4f, acc = %.4f' % (t + 1, loss.data[0], acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(model, loader):

    print('Checking accuracy on validation set')
  
    val_correct = 0
    val_samples = 0
    model.eval()               # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.cuda())

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        val_correct += (preds == y).sum()
        val_samples += preds.size(0)
    acc = float(val_correct) / val_samples
    print('Got %d / %d correct (%.2f)' % (val_correct, val_samples, 100 * acc))
    return acc

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.is_available()

    train_data = TIN200Data('/data1/tiny-imagenet-200', '/data1/tiny-imagenet-200/wnids.txt', data_dir='train')
    val_data = TIN200Data('/data1/tiny-imagenet-200', '/data1/tiny-imagenet-200/wnids.txt', data_dir='val')

    train_loader = DataLoader(train_data, batch_size=192, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=192, shuffle=True, num_workers=2)

    # model = Model().cuda()
    # model = Test_Model().cuda()
    # model = Vgg19().cuda()     # net model in the net.py
    # model = Vgg11().cuda()
    model = GoogleNet().cuda()
    cudnn.benchmark = True

    # model.load_state_dict(torch.load('./net_params/VGG11_net_params.pkl'))
    optimizer = optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-05, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
    best_acc = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        train(model, loss_fn, optimizer, loader=train_loader)
        val_acc = check_accuracy(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
        print('epoch:d%, best_acc:%.2f%%' % (epoch+1, best_acc*100))
        scheduler.step(val_acc, epoch=epoch+1)

    # torch.save(model.state_dict(),'./net_params/GoogleNet_net_params2.pkl')
    torch.save(model, './net_params/GoogleNet.pkl')


if __name__ == '__main__':
    main()