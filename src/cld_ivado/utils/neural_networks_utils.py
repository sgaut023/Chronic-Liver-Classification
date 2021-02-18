import time
import copy
import torch
import numpy as np
import math
import sys
from torchvision import transforms
import torch.nn as nn
import logging 
sys.path.append('../src')
logging.basicConfig(level = logging.INFO)
from cld_ivado.utils.compute_metrics import get_metrics
from cld_ivado.utils.compute_metrics import get_average_per_patient

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, 
                dataset_sizes, num_epochs=5, patience=3, threshold=0.5):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    pre_loss = float('inf') 
    p = patience
    early_stopping = False

    for epoch in range(num_epochs):
        if early_stopping: 
            break
        label_sum = {'positive': 0,'negative': 0, 'cnt': 0}
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
                if phase == 'train': 
                    label_sum['positive'] += sum(labels).item()
                    label_sum['negative'] += labels.shape[0] - sum(labels).item()
                    label_sum['cnt'] += labels.shape[0]
                inputs = inputs.to(device)
                labels = labels.to(device=device, dtype=torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if len(outputs.shape) == 1:
                        preds = (outputs > threshold).float()
                        labels = labels.to(device=device, dtype=torch.float32)
                    else:
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

            if phase == 'val' and pre_loss < epoch_loss and epoch!=0 :
                p -= 1
                if not p:
                    print("Early Stopping")
                    early_stopping = True
                    break
            elif phase == 'val':
                p = patience
                pre_loss = epoch_loss   
                best_model_wts = copy.deepcopy(model.state_dict())   
                print('Saved model')
        print('labels', label_sum)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model

def evaluate_model_metrics(labels, probs, fold_c): 
    acc, auc, specificity, sensitivity = get_metrics(labels, probs)
    (acc_avg, auc_avg, specificity_avg, sensitivity_avg), label_per_patient, average_prob = get_average_per_patient(labels, probs)
       
    if math.isnan(auc):
        logging.info(f'FOLD {fold_c} :  acc: {acc} , specificity: {specificity}, sensitivity: {sensitivity}')
        logging.info(f'FOLD {fold_c} : Average per patient:  acc : {acc_avg} , specificity: {specificity_avg}, sensitivity: {sensitivity_avg}')

    else:
        logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, \
                    specificity: {specificity}, sensitivity: {sensitivity}')
        logging.info(f'FOLD {fold_c} : Average per patient:  acc : {acc_avg} , auc: {auc_avg},\
                    specificity: {specificity_avg}, sensitivity: {sensitivity_avg}')

    test_metric = {'acc': acc, 'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}
    test_metric_avg = {'acc': acc_avg, 'auc': auc_avg, 'sensitivity': sensitivity_avg, 'specificity': specificity_avg}
    return test_metric, test_metric_avg, probs, label_per_patient, average_prob

def get_all_transformations(random_crop_size, is_rgb): 
    if is_rgb is False:
        mean = [0.485]
        std = [0.229]
    else:
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]

    if random_crop_size is None:
        data_transforms = {'train': transforms.Compose([
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),
            'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])}
    else:
        data_transforms = {'train': transforms.Compose([
            transforms.RandomCrop(random_crop_size),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),
            'val': transforms.Compose([
            transforms.CenterCrop(random_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])}
            
    return  data_transforms 


def get_normalize_transformations():
    data_transforms = {'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])]),
        'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])])}
    return  data_transforms 

def get_dataloaders(dataset_train, dataset_val, dataset_test, batch_size ):
    return {'train': torch.utils.data.DataLoader(dataset_train, 
                                                          batch_size=batch_size, 
                                                          shuffle=False, num_workers=4, pin_memory=True),
                       'val': torch.utils.data.DataLoader(dataset_val, 
                                                          batch_size= batch_size, 
                                                          shuffle=False, num_workers=4, pin_memory=True),
                        'test': torch.utils.data.DataLoader(dataset_test, 
                                                          batch_size=batch_size, 
                                                          shuffle=False, num_workers=4,pin_memory=True)}

def create_local_scattering_layers(J, num_components):
        if J==2:
            linear = torch.nn.Linear(num_components * 108 * 159, 1)
            pca_layer = nn.Conv2d(81, num_components, 1)

        elif J==3:
            linear = torch.nn.Linear(num_components * 54 * 79, 1)
            pca_layer = nn.Conv2d(217, num_components, 1)

        elif J==4:
            linear = torch.nn.Linear(num_components * 27 * 39, 1)
            pca_layer = nn.Conv2d(417, num_components, 1)
        
        elif J==5:
            linear = torch.nn.Linear(num_components * 13 * 19, 1)
            pca_layer = nn.Conv2d(681, num_components, 1)
        
        elif J==6:
            linear = torch.nn.Linear(num_components * 6 * 9, 1)
            pca_layer = nn.Conv2d(1009, num_components, 1)
        else:
            raise NotImplemented(f"J {self.J} parameter for scattering not implemented")
            
        return linear, pca_layer
