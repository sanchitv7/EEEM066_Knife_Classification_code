"""import libraries for training"""
import sys

import torch.optim
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
# import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
# from sklearn.model_selection import train_test_split
from data import knifeDataset
from utils import *
import matplotlib.pyplot as plt
import argparse

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

'''Writing the loss and results'''
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()

set_num_workers = get_num_workers()
parser = argparse.ArgumentParser()
parser.add_argument("-mn", "--model-name", required=True)
parser.add_argument("-cn", "--config-name", required=True)

args = parser.parse_args()
model_name = args.model_name
config_name = args.config_name
# log.open("logs/%s_log_train.txt")
# log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
#     datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
# log.write('                           |----- Train -----|----- Valid----|---------|\n')
# log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
# log.write('-------------------------------------------------------------------------------------------\n')


'''Training the model'''


def train(train_loader, model, criterion, optimizer, epoch, valid_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True

    for i, (images, target, fnames) in enumerate(train_loader):
        img = images.to(device, non_blocking=True)
        label = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        print('\r', end='', flush=True)
        message = '%s         %d          %d     |       %0.3f     |       %0.3f     |      %0.3f      |  %s' % (
            "train", i, epoch + 1, losses.avg, valid_accuracy[1], valid_accuracy[0],
            time_to_str((timer() - start), 'sec'))
        print(message, end='', flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]


'''Validating the model'''


def evaluate(val_loader, model, criterion, epoch, train_loss, start):
    losses = AverageMeter()
    model.to(device)
    model.eval()
    model.training = False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.to(device, non_blocking=True)
            label = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            loss = criterion(logits, label)

            losses.update(loss.item(), images.size(0))
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print('\r', end='', flush=True)
            message = '%s            %d          %d     |       %0.3f     |       %0.3f     |      %0.3f      |  %s' % (
                "val", i, epoch + 1, train_loss[0], losses.avg, map.avg, time_to_str((timer() - start), 'sec'))
            print(message, end='', flush=True)
        log.write("\n")
        log.write(message)
    return [map.avg, losses.avg]


'''Computing the mean average precision, accuracy'''


def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


'''------------------ load file and get splits -------------------------'''
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist, mode="train")
train_loader = DataLoader(train_gen,
                          batch_size=config.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=set_num_workers)

val_imlist = pd.read_csv("val.csv")
val_gen = knifeDataset(val_imlist, mode="val")
val_loader = DataLoader(val_gen,
                        batch_size=config.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=set_num_workers)

'''-------------------Loading the model to run----------------------------'''
# Set model
# model_name = 'tf_efficientnet_b0'
model = timm.create_model(model_name, pretrained=True, num_classes=config.n_classes)
model.name = model_name

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

print(f'Using {set_num_workers} workers | Device: {device}\n')

model.to(device)

'''----------------------Parameters--------------------------------------'''
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,
                                           last_epoch=-1)
criterion = nn.CrossEntropyLoss().to(device)

'''------------------------Training---------------------------------------'''
start_epoch = 0
val_metrics = [0, 0]
scaler = torch.cuda.amp.GradScaler()

log.open(f"logs/{model.name}_log_train.txt")
log.write(f'Config: {config_name}')

for k, v in config.__dict__.items():
    log.write(f'{k}: {v}\n')
log.write("\n------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 41))
log.write('                                 |----- Train -----|----- Valid -----|----- Valid -----|------------|\n')
log.write('mode         iter        epoch   |       loss      |       loss      |        mAP      |    time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

if __name__ == '__main__':
    '''train'''
    save_training_losses = []
    save_val_losses = []
    save_val_map = []
    training_start = timer()
    for epoch in range(0, config.epochs):
        lr = get_learning_rate(optimizer)
        start = timer()
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)

        start = timer()
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)

        save_training_losses.append(train_metrics[0])
        save_val_losses.append(val_metrics[1])
        save_val_map.append(val_metrics[0])
        # Saving the model
        if (epoch + 1) % 5 == 0:
            filename = f"{config_name}_Knife-Effb0-E" + str(epoch + 1) + ".pt"
            torch.save(model.state_dict(), filename)

    log.write(f'\n\nTotal time elapsed: {time_to_str(timer() - training_start, mode="sec")}')

    training_losses_tensor = torch.tensor(save_training_losses)
    val_losses_tensor = torch.tensor(save_val_losses)
    val_map_tensor = torch.tensor(save_val_map)

    epochs = range(1, config.epochs + 1)
    epochs_list = list(epochs)  # Convert range object to a list for plt.xticks

    # Plotting training/validation losses vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses_tensor.cpu().numpy(), label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses_tensor.cpu().numpy(), label='Validation Loss', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss vs Epochs')
    plt.xticks(epochs_list)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/content/drive/MyDrive/GitHub Repos/EEEM066_Knife_Classification_code/result_plots/EfficientNet/'
                f'{config_name}/train_val_loss_vs_epochs_{config_name}.png')
    plt.show()

    # Plotting validation mAP vs epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_map_tensor.cpu().numpy(), label='Validation mAP', marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Validation mAP')
    plt.title('Validation mAP vs Epochs')
    plt.xticks(epochs_list)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f'/content/drive/MyDrive/GitHub Repos/EEEM066_Knife_Classification_code/result_plots/EfficientNet/'
        f'{config_name}/val_map_vs_epochs_{config_name}.png')
    plt.show()
