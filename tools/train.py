import os
import torch
import colorama
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from config.config import config
import math

import warnings

warnings.filterwarnings("ignore")

colorama.init(autoreset=True)
epoch_num, weight_decay = config['epoch_num'], config['weight_decay']
initial_lr, max_lr, warmup_epochs = config['initial_lr'], config['max_lr'], config['warmup_epochs']
val_frac = config['val_frac']
save = config['save']
save_per_epoches = config['save_per_epoches']
output_folder = "output"
device = torch.device(config['device'])



def lr_lambda(epoch):
    if epoch < warmup_epochs:
        x = (max_lr / initial_lr) * (epoch + 1) / warmup_epochs
        return x
    else:
        adjusted_epoch = epoch - warmup_epochs
        adjusted_total_epochs = epoch_num - warmup_epochs
        progress = adjusted_epoch / adjusted_total_epochs
        x = max_lr / initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        return x


def evaluate_valid_set(writer, val_loader, model, criterion, epoch):
    loss_list, labels_list = [], []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            loss_list.append(loss)
            positive_probs = outputs[:, 1]
            all_probs.extend(positive_probs)
            labels_list.extend(labels)

    roc_auc = roc_auc_score(labels_list, all_probs)
    print(colorama.Fore.YELLOW + f'AUC: {roc_auc:.5f}', end='\t')
    writer.add_scalar('AUC', roc_auc, epoch)
    return roc_auc


def train(model, loaders, *, binary_weights, log_name='demo'):
    writer = SummaryWriter(f'logs/{log_name}')
    print(colorama.Fore.RED + f'Device: {device}')
    print(colorama.Fore.BLACK + colorama.Back.RED + colorama.Style.BRIGHT + 'tensorboard --logdir=logs')

    train_loader, val_loader = loaders
    print(colorama.Fore.CYAN + f'Train size ~ {len(train_loader) * config["train_batch_size"]}', end='\t')
    print(colorama.Fore.CYAN + f'Valid size ~ {len(val_loader) * config["val_batch_size"]}', end='\t')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=binary_weights)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    batch_idx = 0
    best = 0.
    for epoch in range(epoch_num):
        print(colorama.Fore.GREEN + f'--- Epoch {epoch} ---', end='\t')
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            batch_idx += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(colorama.Fore.BLUE + f'loss: {total_loss / len(train_loader):.5f}', end='\t')
        print(colorama.Fore.MAGENTA + f'lr: {scheduler.get_last_lr()[0]:.6f}', end='\t')
        writer.add_scalar('loss', total_loss, epoch)
        model.eval()
        roc_auc = evaluate_valid_set(writer, val_loader, model, criterion, epoch)

        if save and roc_auc > best:
            if not os.path.exists(f'{output_folder}/{log_name}'):
                os.makedirs(f'{output_folder}/{log_name}')
            torch.save(model.state_dict(), f'{output_folder}/{log_name}/best.pth')
        if save and epoch > 0 and epoch % save_per_epoches == 0:
            if not os.path.exists(f'{output_folder}/{log_name}'):
                os.makedirs(f'{output_folder}/{log_name}')
            torch.save(model.state_dict(), f'{output_folder}/{log_name}/epoch_{epoch}.pth')

        print(colorama.Fore.RED + f'best AUC: {(best := max(roc_auc, best)):.5f}')

    if save:
        if not os.path.exists(f'{output_folder}/{log_name}'):
            os.makedirs(f'{output_folder}/{log_name}')
        torch.save(model.state_dict(), f'{output_folder}/{log_name}/last.pth')
    writer.close()
