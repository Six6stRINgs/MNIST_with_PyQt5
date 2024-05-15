from PyQt5.QtCore import pyqtSignal
import torch
from torch import nn
from torch.optim import Adam, SGD, RMSprop, Adagrad
from torch.utils.data import DataLoader

from networks.MLP import MLP
from networks.AlexNet import AlexNet
from networks.ResNet import ResNet
from networks.ViT import ViT


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean()


def train_model(train_loader: DataLoader, model_config: dict,
                progress_signal: pyqtSignal = None,
                text_signal: pyqtSignal = None,
                loss_signal: pyqtSignal = None
                ) -> tuple[nn.Module, str]:
    optimizer = None
    criterion = None
    model = nn.Module()

    if model_config['model'] == 'MLP':
        model = MLP(num_classes=10, dropout=model_config['dropout'])
    elif model_config['model'] == 'AlexNet':
        model = AlexNet(num_classes=10, dropout=model_config['dropout'])
    elif model_config['model'] == 'ResNet18':
        model = ResNet(num_classes=10, layer=18)
    elif model_config['model'] == 'ResNet34':
        model = ResNet(num_classes=10, layer=34)
    elif model_config['model'] == 'ViT':
        model = ViT(numnum_classes=10, emb_size=16)

    if model_config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=model_config['lr'])
    elif model_config['optimizer'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=model_config['lr'])
    elif model_config['optimizer'] == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=model_config['lr'])

    if model_config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    if model_config['loss'] == 'Focal':
        criterion = FocalLoss()

    model_config_str = 'Model_{}_Epochs_{}_Batch_{}_Lr_{}_Loss_{}_Opt_{}_Dropout_{}_Device_{}_Author_{}'.format(
        model_config['model'],
        model_config['epochs'],
        model_config['batch_size'],
        model_config['lr'],
        model_config['loss'],
        model_config['optimizer'],
        'None' if str(model) == 'ResNet18' or str(model) == 'ResNet34' or str(model) == 'ViT'
        else model_config['dropout'],
        model_config['device'],
        model_config['user']
    )

    epochs = model_config['epochs']
    batch_size = model_config['batch_size']
    device = model_config['device']

    model.train()
    model.to(device)
    total_steps = len(train_loader) * epochs

    if text_signal is not None:
        text = '\nTraining model... \n{}'.format(model_config_str)
        text_signal.emit(text)
        print(text)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if progress_signal is not None:
                current_step = epoch * len(train_loader) + i + 1
                tot = int(current_step / total_steps * 100)
                tot = tot if tot <= 99 else 99
                progress_signal.emit(tot)

            if (i + 1) % batch_size == 0:
                info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    epochs, i + 1,
                    len(train_loader),
                    loss.item()
                )
                print(info)
                if text_signal is not None:
                    text_signal.emit(info)

            if (i + 1) % (batch_size / 4) == 0 and loss_signal is not None:
                loss_signal.emit(loss.item())

    if text_signal is not None:
        text = 'Training finished'
        text_signal.emit(text)
        print(text)
    return model, model_config_str


def test_model(test_loader: DataLoader, model: nn.Module, device: str,
               progress_signal: pyqtSignal = None,
               text_signal: pyqtSignal = None) -> float:
    if text_signal is not None:
        text = '\nStart testing model...'
        text_signal.emit(text)
        print(text)
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total

    res = 'Test Accuracy of the model on the {} test images: {}%'.format(total, acc)
    if progress_signal is not None:
        progress_signal.emit(100)

    if text_signal is not None:
        text_signal.emit(res)
    print(res)
    return acc
