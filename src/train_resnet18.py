import sys
sys.path.append('../src')
import warnings
warnings.filterwarnings("ignore")

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import random
import pandas as pd
from sklearn.model_selection import GroupKFold
from torchvision import transforms

from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics, get_majority_vote,log_test_metrics
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing
from cld_ivado.dataset.dl_dataset import CldIvadoDataset
import copy

def define_model(device, params):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=params['lr'], momentum=params['momentum'])
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=params['step_size'], gamma=params['gamma'])
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=5):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
    


def train_predict():
    catalog, params = get_context()
    dataset = create_dataframe_preproccessing()

    test_metrics={}    
    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed= params['cross_val']['seed']
    fold_c =1 

    df_pid = dataset['id']
    df_y = dataset['labels']

    test_metrics={}  
    #majority vote results
    test_metrics_mv={} 


    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}

    for train_index, test_index in group_kfold_test.split(dataset, df_y, df_pid):
        
        random.seed(seed)
        random.shuffle(train_index)
        X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, criterion, optimizer, scheduler =define_model(device, params['model'])

        dataset_train = CldIvadoDataset(X_train, catalog['data_root'], 'labels', 'fname', data_transforms['train'])
        dataset_test = CldIvadoDataset(X_test, catalog['data_root'], 'labels', 'fname',  data_transforms['val'])

        dataloaders = {'train':torch.utils.data.DataLoader(dataset_train, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=True),
                       'val': torch.utils.data.DataLoader(dataset_test, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=True)}
                            


        model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=5)

        predictions = [int(model.predict(X_test['fname'].iloc[i])[0]) for i in range(X_test.shape[0])]
        
        #get metrics with NO majority vote
        acc, auc, specificity, sensitivity = get_metrics(X_test['labels'], predictions)
        #compute majority vote metrics
        acc_mv, auc_mv, specificity_mv, sensitivity_mv = get_majority_vote(X_test['labels'], np.array(predictions))
        
        print('FOLD '+ str(fold_c) + ':  acc ' + str(acc) +  ', auc ' +  str(auc) +  ', specificity '+ str(specificity)
            + ', sensitivity ' + str(sensitivity))
        print('FOLD '+ str(fold_c) + ':  MV acc ' + str(acc_mv) +  ', MV auc ' +  str(auc_mv) +  ', MV specificity '+ str(specificity_mv)
            + ', MV sensitivity ' + str(sensitivity_mv))
        
        test_metrics[fold_c]=  {'acc':acc, 'auc':auc, 'sensitivity':sensitivity, 'specificity':specificity}
        test_metrics_mv[fold_c]=  {'acc':acc_mv, 'auc':auc_mv, 'sensitivity':sensitivity_mv, 'specificity':specificity_mv}
        
        fold_c +=1 
        
    log_test_metrics(test_metrics, test_metrics_mv, test_n_splits, 'Pretrained Restnet-34 on ImageNet', None, seed)
        

if __name__ =="__main__":
    train_predict()
   
