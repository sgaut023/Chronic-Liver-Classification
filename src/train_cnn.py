import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint
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

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=params['kernel_size'])
        self.conv2 = nn.Conv2d(32, 32, kernel_size=params['kernel_size'])
        self.conv3 = nn.Conv2d(32, 64, kernel_size=params['kernel_size'])
        if params['random_crop_size'] == 448:
            self.fc1 = nn.Linear(774400, 256)
        else:
            self.fc1 = nn.Linear(186624, 256)
        #self.fc1 = nn.Linear(1065088, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = params['dropout']


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view( x.shape[0],  -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def init_weights(m):
    if type(m) == nn.Conv2d:
        #torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)

def define_model(device, params):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
    if params['name'] == 'vgg-16':
        model_ft = models.vgg16(pretrained=params['pretrained'])
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 2)
    
    elif params['name'] == 'resnet-18':
        model_ft = models.resnet34(pretrained=params['pretrained'])
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
    
    elif params['name'] == 'cnn':
        model_ft = Net(params)
        model_ft = model_ft.to(device)
        model_ft.apply(init_weights)

    else:
        raise NotImplemented(f"Model {params['name']} not implemented")

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if params['optimizer'] == 'sgd':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=params['lr'])#, momentum= params['momentum'])
    elif params['optimizer'] == 'adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=params['lr'])
    else:
        raise NotImplemented(f"Optimizer {params['optimizer']} not implemented")
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=params['step_size'], gamma=params['gamma'])
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=5, patience =3):
    # from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    pre_loss = float('inf') 
    p= patience
    early_stopping = False
    for epoch in range(num_epochs):
        if early_stopping: break
        label_sum={'positive':0,'negative':0, 'cnt':0}
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
                    prob = torch.nn.functional.softmax(outputs, dim=1)
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
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, criterion, device, fold_c):
    # from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#5017
    test_losses = []
    logits = []
    predictions = []
    labels = []
    label_sum={'positive':0,'negative':0, 'cnt':0,'p_list':[], 'pred_list':[], 'outputs': []}
    with torch.no_grad():
        for x_test, y_test in test_loader:
            label_sum['positive'] += sum(y_test).item()
            label_sum['negative'] += y_test.shape[0] - sum(y_test).item()
            label_sum['cnt'] += y_test.shape[0]
            label_sum['p_list'].extend(y_test.numpy())
            x_test = x_test.to(device)
            y_test = y_test.to(device, dtype=torch.int64)                   
            model.eval()
            outputs = model(x_test)
            prob = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            test_loss = criterion(outputs, y_test)
            label_sum['pred_list'].extend(preds.cpu().numpy())
            label_sum['outputs'].extend(prob.cpu().numpy()[:, 1])

            #logits.append(outputs)
            predictions.extend(preds.cpu().detach().numpy())
            test_losses.append(test_loss.item())
            labels.extend(y_test.cpu().detach().numpy())
            logits.extend(prob[:, 1].cpu().detach().numpy())
        print('labels', label_sum)

    # get metrics with NO majority vote
    acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
    # compute majority vote metrics
    acc_mv, auc_mv, specificity_mv, sensitivity_mv = get_majority_vote(labels, predictions)
    
    logging.info(f'FOLD {fold_c} :  test_loss_avg: {np.array(test_losses).mean()}')
    #logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, specificity: {specificity}, sensitivity: {sensitivity}')
    logging.info(f'FOLD {fold_c} :  auc: {auc}')
    #logging.info(f'FOLD {fold_c} :  acc_mv: {acc_mv} , auc_mv: {auc_mv}, specificity_mv: {specificity_mv}, sensitivity_mv: {sensitivity_mv}')

    test_metric=  {'acc':acc, 'auc':auc, 'sensitivity':sensitivity, 'specificity':specificity}
    test_metric_mv=  {'acc':acc_mv, 'auc':auc_mv, 'sensitivity':sensitivity_mv, 'specificity':specificity_mv}
    return test_metric, test_metric_mv


def train_predict(catalog, params):
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
        transforms.RandomResizedCrop(params['model']['random_crop_size']),
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.485], [0.229])
    ]),'val': transforms.Compose([
        transforms.CenterCrop(params['model']['random_crop_size']),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.485], [0.2295])
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

        dataset_train = CldIvadoDataset(subtrain_data, catalog['data_root'], 'labels', 'fname', 
                                        data_transforms['train'], is_rgb = False)
        dataset_val = CldIvadoDataset(val_data, catalog['data_root'], 'labels', 'fname',  
                                        data_transforms['val'],  is_rgb = False)
        dataset_test = CldIvadoDataset(X_test, catalog['data_root'], 'labels', 'fname', 
                                        data_transforms['val'],  is_rgb = False)

        dataloaders = {'train': torch.utils.data.DataLoader(dataset_train, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=True),
                       'val': torch.utils.data.DataLoader(dataset_val, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False),
                        'test': torch.utils.data.DataLoader(dataset_test, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False)}
        logging.info(f'FOLD {fold_c}: model train started')
        # start training
        dataset_sizes = {'train': len(subtrain_data), 'val':  len(val_data)}
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=params['model']['epoch'],
                            patience=params['model']['patience'])
        logging.info(f'FOLD {fold_c}: model train done')

        # model evaluation
        test_metric, test_metric_mv = evaluate_model(model, dataloaders['test'], criterion, device, fold_c)
        test_metrics[fold_c] =  test_metric
        test_metrics_mv[fold_c] =  test_metric_mv     
        fold_c +=1 

    log_test_metrics(test_metrics, test_metrics_mv, params)
       
        
if __name__ =="__main__":
    catalog, params = get_context()
    # # Step 3: Define a random search for these parameters, for hyperparameter tuning
    # #        random_number_generator = np.random.RandomState(0) 
    # param_grid = {'lr': uniform(loc=0.00001, scale=0.001),'optimizer': ['sgd', 'adam'],
    #                  'pretrained': [False, True],'batch_size': [32,64]}#, 'dropout': uniform(loc=0.01, scale=0.7)}
    # param_list = list(ParameterSampler(param_grid, n_iter=params['model']['search_iter'], 
    #                                 random_state=params['cross_val']['seed']))
    
    # # Perform hyperparameter search
    # for param_dict in param_list:
    #     print(f"Hyperparams: optimizer= {param_dict['optimizer']}, lr= {round(param_dict['lr'],5)},  \
    #         pretrained: {param_dict['pretrained']}, 'batch_size: {param_dict['batch_size']}")#, dropout: {param_dict['dropout']}")
    #     params['model']['optimizer'] = param_dict['optimizer']
    #     params['model']['lr'] = round(param_dict['lr'], 5)
    #     params['model']['pretrained'] = param_dict['pretrained']
    #     params['model']['batch_size'] = param_dict['batch_size']
    #     #params['model']['dropout'] = param_dict['dropout']
    #     train_predict(catalog, params) 
    train_predict(catalog, params) 
