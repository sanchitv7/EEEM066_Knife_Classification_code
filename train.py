"""import libraries for training"""
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
from pprint import pformat

warnings.filterwarnings('ignore')

'''Writing the loss and results'''
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
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
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % ( \
            "train", i, epoch, losses.avg, valid_accuracy[0], time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]


'''Validating the model'''


def evaluate(val_loader, model, criterion, epoch, train_loss, start):
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

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (
                "val", i, epoch, train_loss[0], map.avg, time_to_str((timer() - start), 'min'))
            print(message, end='', flush=True)
        log.write("\n")
        log.write(message)
    return [map.avg]


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
train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_imlist = pd.read_csv("val.csv")
val_gen = knifeDataset(val_imlist, mode="val")
val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8)

'''-------------------Loading the model to run----------------------------'''
model_name = 'tf_efficientnet_b0'
model = timm.create_model(model_name, pretrained=True, num_classes=config.n_classes)
model.name = model_name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
model.to(device)
print(f'Using backend: {device}')

'''----------------------Parameters--------------------------------------'''
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,
                                           last_epoch=-1)
criterion = nn.CrossEntropyLoss().to(device)

'''------------------------Training---------------------------------------'''
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()

if __name__ == '__main__':
    '''train'''

    log.open(f"logs/{model.name}_log_train.txt")
    # log.write(pformat(config.__dict__))
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('                           |----- Train -----|----- Valid----|---------|\n')
    log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
    log.write('-------------------------------------------------------------------------------------------\n')
    for epoch in range(0, config.epochs):
        lr = get_learning_rate(optimizer)
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
        # Saving the model
        if (epoch + 1) % 5 == 0:
            filename = "Knife-Effb0-E" + str(epoch + 1) + ".pt"
            torch.save(model.state_dict(), filename)
