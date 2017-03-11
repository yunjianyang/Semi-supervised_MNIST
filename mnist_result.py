from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy import ndimage

# Training settings
parser = argparse.ArgumentParser(description = 'MNIST Parameters')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default:0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default:1)')
parser.add_argument('--dropout_rate', type=float, default=0.5, metavar='DO',
                    help='dropout rate (default: 0.5)')
parser.add_argument('--output_file', type=str, default='output.csv', metavar='OF',
                    help='output file (default: output.csv)')
parser.add_argument('--batch_norm', type=int, default=1, metavar='BN',
                    help='batch nomralization (default: 1)')
parser.add_argument('--ada_delta', type=int, default=1, metavar='AD',
                    help='adaptive learning rate (default: 1)')
parser.add_argument('--data_aug', type=int, default=1, metavar='DA',
                    help='data augmentation (default: 1)')
parser.add_argument('--init_round', type=int, default=15, metavar='IR',
                    help='initialization round (default: 15)')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d(p = args.dropout_rate)
        self.batch_norm_1 = nn.BatchNorm2d(1)
        self.batch_norm_2 = nn.BatchNorm2d(50)
        self.batch_norm_3 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(7 * 7 * 50, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        if args.batch_norm:
            x = self.batch_norm_1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        if args.batch_norm:
            x = self.batch_norm_2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        if args.batch_norm:
            x = self.batch_norm_3(x)
        x = x.view(args.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = torch.load('output.cnn_model')

validset = pickle.load(open("validation.p", "rb"))
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers = 2)
epoch = 0

model.eval()
test_loss = 0
correct = 0
for data, target in valid_loader:
    #data = conv_remap.remap(data, centers)
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss += F.nll_loss(output, target).data[0]
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()
test_loss /= len(valid_loader) # loss function already averages over batch size
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(valid_loader.dataset),
    100. * correct / len(valid_loader.dataset)))


f = open('output.csv', 'w')
f.write('ID,label\n')
testset = pickle.load(open("test.p", "rb"))
testset.train_labels = torch.Tensor(len(testset)).fill_(-1)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers = 2)
model.eval()
wrt_idx = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    pred = output.data.max(1)[1] 
    for i in range(len(pred)):
        f.write(str(wrt_idx))
        f.write(',')
        f.write(str(pred.view(-1)[i]))
        f.write('\n')
        wrt_idx += 1
f.close()
