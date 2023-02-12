import os
import sys
import argparse
from datasets import batch_size, num_classes, channels
from datasets import class_list, train_loader, test_loader
from helper import train_model, save_model
from helper import plot_training_loss, plot_accuracy
from helper import compute_confusion_matrix, plot_confusion_matrix
import torch
import torch.nn


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', default='hymenoptera_data',
                    type=str, required=True, help='Path to the data folder.')
parser.add_argument('-o', '--outputs_path', type=str,
                    required=True, help='Path to the output folder: \
                    model, plots, and confusion matrix.')

args = parser.parse_args()

if not os.path.exists(args.outputs_path):
    os.makedirs(args.outputs_path)

if not os.path.isdir(args.input_path):
    print('The input path specified does not exist')
    sys.exit()
print('\n'.join(os.listdir(args.input_path)))


# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# hyperparameters
num_epochs = 20
batch_size = batch_size
num_classes = num_classes
in_channels = channels
lr = 1e-4
momentum = 0.9


# ResNet18
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3,
                           stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1,
                           stride=stride, bias=False)


class BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(torch.nn.Module):

    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7,
                                     stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                torch.nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# initialization
model = ResNet18(BasicBlock, layers=[2, 2, 2, 2],
                 num_classes=num_classes).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)


train_losses, train_acc_list = train_model(model, num_epochs, loss,
                                           optimizer, train_loader,
                                           test_loader, device)


# save:
# trained model weights
save_model(num_epochs, model, optimizer, loss,
           name='resnet18')
# loss and accuracy plots
plot_training_loss(train_losses, name='loss_resnet18',
                   title='Training Loss: ResNet18')
plot_accuracy(train_acc_list, name='accuracy_resnet18',
              title='Training Accuracy: ResNet18')
# confusion matrix
mat = compute_confusion_matrix(model, test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_list, name='confusionM_resnet18',
                      title='Confusion Matrix: ResNet18')
