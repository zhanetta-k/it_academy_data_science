import os
import sys
import argparse
from datasets import batch_size, num_classes, channels
from datasets import class_list, train_loader_aug, test_loader
from helper import train_model, plot_training_loss, plot_accuracy
from helper import compute_confusion_matrix, plot_confusion_matrix
import torch
import torchvision


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path',
                    default='hymenoptera_data/hymenoptera_data',
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


# ResNet18
model = torchvision.models.resnet18(pretrained=True).to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(in_features=512, out_features=classes)

print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad is True:
        params_to_update.append(param)
        print("\t", name)


# training
train_losses, train_acc_list = train_model(model, num_epochs, loss,
                                           optimizer, train_loader_aug,
                                           test_loader, device)


# save:
# loss and accuracy plots
plot_training_loss(train_losses, name='loss_resnet18_tuned_aug',
                   title='Training Loss: ResNet18_tuned_aug')
plot_accuracy(train_acc_list, name='accuracy_resnet18_tuned_aug',
              title='Training Accuracy: ResNet18_tuned_aug')
# confusion matrix
mat = compute_confusion_matrix(model, test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_list,
                      name='confusionM_resnet18_tuned_aug',
                      title='Confusion Matrix: ResNet18_tuned_aug')
