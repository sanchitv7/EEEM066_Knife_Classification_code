"""import libraries for training"""
import warnings
# from datetime import datetime
# from timeit import default_timer as timer
import pandas as pd
import torch.optim
# from sklearn.model_selection import train_test_split
# from torch import optim
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-mn", "--model-name", required=True)
parser.add_argument('-mw', '--model-weights', required=True)
parser.add_argument("-cn", "--config-name", required=True)


args = parser.parse_args()
model_weights_to_use = args.model_weights
model_name_to_use = args.model_name
config_name = args.config_name


warnings.filterwarnings('ignore', category=Warning)

# log = Logger()

'''Validating the model'''


def evaluate(val_loader, model):
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
    return map.avg


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


'''------------------------------load file and get splits--------------------------------------------'''
# Choose model
# model_name_to_use = 'tf_efficientnet_b0'
# model_weights_to_use = 'Conf_6_Knife-Effb0-E10.pt'


# Create an instance of Logger
log = Logger()

# Open the log file in append mode
log.open(f'logs/{model_name_to_use}_log_train_{config_name}.txt', mode='a')
# Set device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

set_num_workers = get_num_workers()
log.write(f'\n------------Test Results---------------\n')
log.write(f'\nUsing {set_num_workers} workers | Device: {device}\n')

log.write('\nreading test file')
test_files = pd.read_csv("test.csv")
log.write('\nCreating test dataloader')
test_gen = knifeDataset(test_files, mode="test")
test_loader = DataLoader(test_gen, batch_size=64, shuffle=False, pin_memory=True, num_workers=set_num_workers)

log.write('\nloading trained model')
model = timm.create_model(model_name_to_use, pretrained=True, num_classes=config.n_classes)
model.name = model_name_to_use
model.load_state_dict(torch.load(model_weights_to_use, map_location=torch.device(device)))
model.to(device)


'''-----------------------------------------Testing-----------------------------------------------'''
if __name__ == '__main__':
    log.write('\nEvaluating trained model')
    map = evaluate(test_loader, model)
    # log.open(f"logs/{model.name}_log_train.txt")

    log.write(f"Testing results for: {model_weights_to_use}")
    log.write(f"\nmAP = {map.item():.3f}")
