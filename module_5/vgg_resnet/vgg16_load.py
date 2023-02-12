import os
import sys
import argparse
from datasets import batch_size, num_classes, channels
from datasets import class_list, train_loader, test_loader
from helper import train_model, plot_training_loss, plot_accuracy
from helper import compute_confusion_matrix, plot_confusion_matrix
from vgg16 import VGG16
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
classes = num_classes
in_channels = channels
lr = 1e-4
momentum = 0.9


# initialize the model and load the trained weights
model = VGG16(in_channels=in_channels, num_classes=classes).to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

# load state
checkpoint = torch.load('outputs/vgg16.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_losses, train_acc_list = train_model(model, num_epochs, loss,
                                           optimizer, train_loader,
                                           test_loader, device)


# save:
# loss and accuracy plots
plot_training_loss(train_losses, name='loss_vgg16_load',
                   title='Training Loss: VGG16_load')
plot_accuracy(train_acc_list, name='accuracy_vgg16_load',
              title='Training Accuracy: VGG16_load')
# confusion matrix
mat = compute_confusion_matrix(model, test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_list,
                      name='confusionM_vgg16_load',
                      title='Confusion Matrix: VGG16_load')
