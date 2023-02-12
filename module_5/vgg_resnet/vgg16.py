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


# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# hyperparameters
num_epochs = 20
batch_size = batch_size
classes = num_classes
in_channels = channels
lr = 1e-4
momentum = 0.9


# VGG16 Model
VGG16_arc = [64, 64, "M",
             128, 128, "M",
             256, 256, 256, "M",
             512, 512, 512, "M",
             512, 512, 512, "M"]


class VGG16(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(VGG16_arc)

        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, num_classes))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for i in architecture:
            if type(i) == int:
                out_channels = i
                layers += [torch.nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=1),
                           torch.nn.BatchNorm2d(i),
                           torch.nn.ReLU(inplace=True)]
                in_channels = i
            elif i == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=(2, 2),
                                              stride=(2, 2))]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # x = x.reshape(x.shape[0], -1)
        logits = self.fcs(x)
        return logits


# initialization
model = VGG16(in_channels=in_channels, num_classes=classes).to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)


train_losses, train_acc_list = train_model(model, num_epochs, loss,
                                           optimizer, train_loader,
                                           test_loader, device)


# save the trained model weights
save_model(num_epochs, model, optimizer, loss,
           name='vgg16')
# save the loss and accuracy plots
plot_training_loss(train_losses, name='loss_vgg16',
                   title='Training Loss: VGG16')
plot_accuracy(train_acc_list, name='accuracy_vgg16',
              title='Training Accuracy: VGG16')
# save confusion matrix
mat = compute_confusion_matrix(model, test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_list, name='confusionM_vgg16',
                      title='Confusion Matrix: VGG16')
