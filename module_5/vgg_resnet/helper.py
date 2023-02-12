import datetime
import numpy as np
import torch
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion,
               name='model'):
    """
    Save the trained model.
    """
    torch.save({'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/{}.pth'.format(name))


def train_model(model, num_epochs, loss, optimizer,
                train_loader, test_loader, device):
    start_time = datetime.datetime.now()
    train_losses = np.zeros(num_epochs)
    train_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        t0 = datetime.datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            losses = loss(outputs, targets)
            optimizer.zero_grad()

            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())

        train_loss = np.mean(train_loss)
        train_losses[epoch] = train_loss

        dt = datetime.datetime.now() - t0
        print(f'Epoch: {epoch+1}/{num_epochs} \
              | Time elapsed: {dt} | Train Loss: {train_loss:.4f}')

        model.eval()
        # save memory during inference
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device=device)
            print(f'Train accuracy: {train_acc :.2f}% ')
            train_acc_list.append(train_acc.item())

    elapsed = datetime.datetime.now() - start_time
    print(f'Total Training Time: {elapsed}')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')
    return train_losses, train_acc_list


def plot_training_loss(train_losses, name='loss',
                       title='Training Loss'):
    plt.plot(train_losses)
    plt.ylim([0.40, 0.80])
    plt.xlim([0, 20])
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Minibatch Loss', fontsize=11)
    plt.legend(['training loss'])
    plt.title('{}'.format(title), fontsize=15)
    plt.tight_layout()
    plt.savefig('outputs/{}.png'.format(name))


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def plot_accuracy(train_acc_list, name='accuracy',
                  title='Training Accuracy'):
    # num_epochs = len(train_acc_list)

    # plt.plot(np.arange(1, num_epochs+1),
    # train_acc_list, label='Training')
    plt.plot(train_acc_list)
    plt.ylim([0, 100])
    plt.xlim([0, 20])
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('{}'.format(title), fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/{}.png'.format(name))


def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat


def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=(6, 6),
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None,
                          title='Confusion Matrix',
                          name='confusion_matrix'):

    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat)*1.10, len(conf_mat)*1.10)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label', fontsize=11)
    plt.ylabel('true label', fontsize=11)
    plt.title('{}'.format(title), loc='center', fontsize=15)
    plt.savefig('outputs/{}.png'.format(name))


def show_examples(model, data_loader, class_list=None,
                  name='pred_examples'):
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplots(nrows=3, ncols=5,
                             sharex=True, sharey=True)

    for idx in range(features.shape[0]):
        features[idx] = features[idx] / 2 + 0.5  # unnormalize
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')
            if class_list is not None:
                ax.title.set_text(f'P: {class_list[predictions[idx]]}'
                                  f'\nT: {class_list[targets[idx]]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_list is not None:
                ax.title.set_text(f'P: {class_list[predictions[idx]]}'
                                  f'\nT: {class_list[targets[idx]]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False
    plt.tight_layout()
    plt.savefig('outputs/[{}].png'.format(name))
