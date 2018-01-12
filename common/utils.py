'''
@file: utils.py
@version: v1.0
@date: 2018-01-07
@author: ruanxiaoyi
@brief: Common utils
@remark: {when} {email} {do what}
'''

import time
import torch
from torch.autograd import Variable

def localtime():
    '''
    Get current time
    '''
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def save(net, state_dict=False, replace=False):
    '''
    Save a network
    '''
    if replace:
        if state_dict:
            torch.save(net.state_dict(), f'./saved_nets/best_state.pkl')
        else:
            torch.save(net, f'./saved_nets/best.pkl')
    else:
        if state_dict:
            torch.save(net.state_dict(), f'./saved_nets/net_state_{localtime()}.pkl')
        else:
            torch.save(net, f'./saved_nets/net_{localtime()}.pkl')


def restore(pkl_path, model_class=None):
    '''
    Restore a network
    '''
    if model_class != None:
        try:
            model = model_class()
            return model.load_state_dict(torch.load(pkl_path))
        except:
            raise ValueError('model_class must match with the model you want to restore')

    else:
        return torch.load(pkl_path)


def check_accuracy(net, loader):
    '''
    Check the accuracy of val data
    '''
    print('Checking accuracy on validation set')

    num_correct = 0
    num_samples = 0
    # Put the net in test mode (the opposite of net.train(), essentially)
    net.eval()
    for x, y in loader:
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_var = Variable(x, volatile=True)

        scores = net(x_var.cuda())
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = 100.0 * float(num_correct) / num_samples
    print(f'Got {num_correct} / {num_samples} correct ({acc:.2f}%)')
    return acc
