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
from scipy import misc

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
parser.add_argument('--output_file', type=str, default='output', metavar='OF',
                    help='output file (default: output)')
parser.add_argument('--batch_norm', type=int, default=1, metavar='BN',
                    help='batch nomralization (default: 1)')
parser.add_argument('--ada_delta', type=int, default=1, metavar='AD',
                    help='adaptive learning rate (default: 1)')
parser.add_argument('--data_aug', type=int, default=1, metavar='DA',
                    help='data augmentation (default: 1)')
parser.add_argument('--init_round', type=int, default=15, metavar='IR',
                    help='initialization round (default: 15)')
args = parser.parse_args()

# Loading data
print('loading data!')
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
validset = pickle.load(open("validation.p", "rb"))

#data augmentation
'''
if args.data_aug:
    new_data = trainset_labeled.train_data.clone()
    new_data_labels = trainset_labeled.train_labels.clone()
    new_data.numpy()[:] = ndimage.rotate(new_data.numpy(), 0.2, reshape = False);
    trainset_labeled.train_data = torch.cat((trainset_labeled.train_data, new_data), 0)
    trainset_labeled.train_labels = torch.cat((trainset_labeled.train_labels, new_data_labels), 0)
    trainset_labeled.k = 6000
'''
def data_augment(s):
    data = s.train_data
    label = s.train_labels
    new_data = np.ndarray(shape=(0,28,28), dtype='uint8')
    new_label = torch.LongTensor(0,1)
    n = 6
    for i in range(len(s)):
        for k in range(1,n+1):
            tmp = data[i].numpy()
            new1 = misc.imrotate(tmp, k*5)
            new2 = misc.imrotate(tmp, 360-(k*5))
            new_data = np.append(new_data, [new1, new2], axis=0)
        new_label = torch.cat((new_label, torch.LongTensor(2*n,1).fill_(label[i])), 0)
    s.train_data = torch.cat((s.train_data, torch.from_numpy(new_data)), 0)
    s.train_labels = torch.cat((s.train_labels, new_label), 0).view(-1)
    s.k = len(s.train_labels)
    return s

if args.data_aug:
    trainset_labeled = data_augment(trainset_labeled)


train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=args.batch_size, shuffle=True, num_workers = 2)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers = 2)


trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
trainset_unlabeled.train_labels = torch.Tensor(len(trainset_unlabeled)).fill_(-1)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=50, shuffle=True, num_workers = 2)




# Architecture of Convolutional Neural Net
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

model = Net()

# optimization method
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
if args.ada_delta:
    optimizer = optim.Adadelta(model.parameters())

# training
def train(epoch):
    model.train()
    
    #pre-training without unlabeled data
    if epoch > args.init_round:
        for batch_idx, (data, target) in enumerate(train_unlabeled_loader):
            model.eval()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            fake_target = Variable(output.data.max(1)[1].view(-1)) # using pseudo label method and get pseudo labels
            model.train()
            data.volatile = False
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, fake_target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_unlabeled_loader.dataset),
                    100. * batch_idx / len(train_unlabeled_loader), loss.data[0]))

    avg_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = conv_remap.remap(data, centers)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            avg_train_loss += loss.data[0]
    avg_train_loss = avg_train_loss / (trainset_labeled.k / 500)
    loss_compare.write(str(avg_train_loss) + ',')


def test(epoch, valid_loader, best_rate, best_epoch, model_name):
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
    if best_rate < (100. * correct / len(valid_loader.dataset)):
        best_rate = 100. * correct / len(valid_loader.dataset)
        best_epoch = epoch
        torch.save(model, model_name)
    loss_compare.write(str(test_loss) + '\n')
    accuracy_compare.write(str(100. * correct / len(valid_loader.dataset)) + '\n')
    return best_rate

loss_compare = open(args.output_file + 'loss_comparasion.csv', 'w')
loss_compare.write('Train_Loss,Validation_Loss\n')

accuracy_compare = open(args.output_file + 'accuracy.csv', 'w')
accuracy_compare.write('Validation_accuracy\n')

best_rate = 0
best_epoch = 0
model_name = args.output_file + '.cnn_model'

for epoch in range(1, args.epochs + 1):
    train(epoch)
    best_rate = test(epoch, valid_loader, best_rate, best_epoch, model_name)
    print(best_rate)
loss_compare.close()
accuracy_compare.close()
print('best rate: ')
print(str(bset_rate) + '\n')
print('best epoch: ')
print(str(best_epoch) + '\n')
