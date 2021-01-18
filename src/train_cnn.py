import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
import time
import os
import random
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold

from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics, get_majority_vote,log_test_metrics
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing
from cld_ivado.utils.split import train_test_split
from cld_ivado.dataset.dl_dataset import CldIvadoDataset
import copy

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)

# from: https://nextjournal.com/gkoehler/pytorch-mnist
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(104*155*64, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,104*155*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x

def define_model(device, params):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model_ft = Net()
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=params['lr'], momentum= params['momentum'])
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=params['lr'])
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=params['step_size'], gamma=params['gamma'])
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=5):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        #for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device=device, dtype=torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, criterion, device, fold_c):
    # from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#5017
    test_losses = []
    logits = []
    predictions = []
    labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device, dtype=torch.int64)                   
            model.eval()
            outputs = model(x_test)
            _, preds = torch.max(outputs, 1)
            test_loss = criterion(outputs, y_test)

            #logits.append(outputs)
            predictions.extend(preds.cpu().detach().numpy())
            test_losses.append(test_loss.item())
            labels.extend(y_test.cpu().detach().numpy())
            logits.extend(outputs[:, 1].cpu().detach().numpy())

    # get metrics with NO majority vote
    acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
    # compute majority vote metrics
    acc_mv, auc_mv, specificity_mv, sensitivity_mv = get_majority_vote(labels, predictions)
    
    logging.info(f'FOLD {fold_c} :  test_loss_avg: {np.array(test_losses).mean()}')
    logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, specificity: {specificity}, sensitivity: {sensitivity}')
    logging.info(f'FOLD {fold_c} :  acc_mv: {acc_mv} , auc_mv: {auc_mv}, specificity_mv: {specificity_mv}, sensitivity_mv: {sensitivity_mv}')

    test_metric=  {'acc':acc, 'auc':auc, 'sensitivity':sensitivity, 'specificity':specificity}
    test_metric_mv=  {'acc':acc_mv, 'auc':auc_mv, 'sensitivity':sensitivity_mv, 'specificity':specificity_mv}
    return test_metric, test_metric_mv


def train_predict():

    catalog, params = get_context()
    dataset = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    dataset = create_dataframe_preproccessing(dataset)
  
    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c =1 


    df_pid = dataset['id']
    df_y = dataset['labels']

    test_metrics ={}  
    test_metrics_mv ={}     #majority vote results

    data_transforms = {'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),'val': transforms.Compose([
       # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Cross-validation Started')
    for train_index, test_index in group_kfold_test.split(dataset, df_y, df_pid):

        random.seed(seed)
        random.shuffle(train_index)
        X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
        model, criterion, optimizer, scheduler = define_model(device, params['model'])

        # split training set in subtrain and validation set
        subtrain_data, val_data = train_test_split(X_train, train_sz=params['model']['train_pct'], seed=seed)

        dataset_train = CldIvadoDataset(subtrain_data, catalog['data_root'], 'labels', 'fname', data_transforms['train'], False)
        dataset_val = CldIvadoDataset(val_data, catalog['data_root'], 'labels', 'fname',  data_transforms['val'], False)
        dataset_test = CldIvadoDataset(X_test, catalog['data_root'], 'labels', 'fname', data_transforms['val'], False)

        dataloaders = {'train': torch.utils.data.DataLoader(dataset_train, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False),
                       'val': torch.utils.data.DataLoader(dataset_val, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False),
                        'test': torch.utils.data.DataLoader(dataset_test, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False)}
        logging.info(f'FOLD {fold_c}: model train started')
        # start training
        dataset_sizes = {'train': len(subtrain_data), 'val':  len(val_data)}
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=params['model']['epoch'])
        logging.info(f'FOLD {fold_c}: model train done')

        # model evaluation
        test_metric, test_metric_mv = evaluate_model(model, dataloaders['test'], criterion, device, fold_c)
        test_metrics[fold_c] =  test_metric
        test_metrics_mv[fold_c] =  test_metric_mv     
        fold_c +=1 

    log_test_metrics(test_metrics, test_metrics_mv, params)
       
        
if __name__ =="__main__":
    train_predict() 
