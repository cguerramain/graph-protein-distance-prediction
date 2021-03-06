import torch
import torch.nn as nn
import pickle
from collections import defaultdict
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, log_interval=100, max_iter=float('Inf')):
    model.train()
    total_iters = min(max_iter, len(train_loader))

    model = model.to(device, non_blocking=True)
    running_loss = 0.0
    for batch_idx, (features, labels, indices) in tqdm(enumerate(train_loader), total=total_iters):
        if batch_idx >= max_iter:
            break
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            return outputs, float(loss.item())

        outputs, batch_loss = handle_batch()
        running_loss += batch_loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), batch_loss))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, (batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset),
               100. * (batch_idx + 1) / len(train_loader), batch_loss))
    return running_loss / len(train_loader)


def test(model, device, test_loader, criterion, epoch):
    if not device:
        device = get_default_device()
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        model = model.to(device, non_blocking=True)
        for batch_idx, (features, labels, indices) in tqdm(enumerate(test_loader), total=len(test_loader)):
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # forward + backward + optimize
            def handle_batch():
                """Function done to ensure variables immediately get dealloced"""
                outputs = model(features)
                loss = criterion(outputs, labels)
                return outputs, float(loss.item())

            outputs, batch_loss = handle_batch()
            running_loss += batch_loss
        avg_loss = running_loss / len(test_loader)
        print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch, avg_loss))
        return avg_loss


def train_and_validate(model, train_loader, test_loader, lr=1e-5, device=None, epochs=10, class_weights=None,
                       save_file=None, **kwargs):
    if not device:
        device = get_default_device()
    if not save_file:
        save_file = '{}_{}.p'.format(model.__class__.__name__, datetime.now().strftime('%d-%m-%y_%H:%M:%S'))
    if class_weights is not None:
        class_weights = class_weights.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input_ = train_loader.dataset[0][0]
    model = model.to(device, non_blocking=True)
    summary(model, input_size=input_.shape)
    losses = defaultdict(list)
    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch + 1, **kwargs)
        test_loss = test(model, device, test_loader, criterion, epoch + 1)

        model_file = '{}_epoch{}.{}'.format(save_file.split('.')[0], epoch + 1, save_file.split('.')[1])
        loss_file = '{}_epoch{}_loss.{}'.format(save_file.split('.')[0], epoch + 1, save_file.split('.')[1])
        torch.save(model.state_dict(), model_file)
        losses['training_losses'].append(train_loss)
        losses['test_losses'].append(test_loss)
        pickle.dump(losses, open(loss_file, 'wb'))


def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """Counts the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

