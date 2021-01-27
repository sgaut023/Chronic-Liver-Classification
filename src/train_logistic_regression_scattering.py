import sys
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint
import numpy as np
from torchvision import transforms
import time
import os
import random
import pandas as pd
import logging
from kymatio.torch import Scattering2D
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics, get_majority_vote,log_test_metrics
from cld_ivado.utils.split import get_train_test_patients_id
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing
from cld_ivado.dataset.dl_dataset import CldIvadoDataset,CldIvadoEntireDataset
import copy

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_components, pca, random_crop_size=None, is_scattering=False, J=2):
        super(LogisticRegression, self).__init__()            
        if random_crop_size is None:
            #self.pca_layer = torch.nn.Linear(434 * 636, num_components)
            self.pca_layer = torch.nn.Linear(81 * 108 * 159, num_components)
        else: 
            self.pca_layer = torch.nn.Linear(random_crop_size * random_crop_size, num_components)
        
        self.pca_layer.weight.data = torch.from_numpy(pca.components_[0:num_components])
        self.linear = torch.nn.Linear(num_components, 1)
        self.is_scattering = is_scattering
        self.J = J
        
        #logistic regression - initialize weights and bias to 0
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
      
    def forward(self, x):
        with torch.no_grad():
            # if self.is_scattering:
            #     use_cuda = torch.cuda.is_available()
            #     S = Scattering2D(self.J,(434, 636))
            #     if use_cuda: S = S.cuda()
            #     x = S.scattering(x)
            pca_components = self.pca_layer(x.view(x.shape[0],-1))

        y_pred = F.sigmoid(self.linear(pca_components))
        return y_pred.view(-1)


def define_model(device, params, num_components, pca, is_scattering = False, J=2, random_crop_size = None):
    model_ft = LogisticRegression(num_components= num_components, pca = pca,
                                    is_scattering = is_scattering , J=J ,
                                    random_crop_size = random_crop_size)
    model_ft = model_ft.to(device)
    criterion = nn.BCELoss(size_average=True)

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


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, 
                num_epochs=5, patience =3, threshold = 0.5):
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
                labels = labels.to(device=device, dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > threshold).float()
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

            epoch_loss = round(running_loss / dataset_sizes[phase],4)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and pre_loss <= epoch_loss and epoch!=0 :
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

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, criterion, device, fold_c, threshold = 0.5):
    # from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#5017
    test_losses = []
    logits = []
    predictions = []
    labels = []
    label_sum = {'positive': 0,'negative': 0, 'cnt': 0,'p_list': [], 'pred_list': [], 'outputs':[]}
    with torch.no_grad():
        for x_test, y_test in test_loader:
            label_sum['positive'] += sum(y_test).item()
            label_sum['negative'] += y_test.shape[0] - sum(y_test).item()
            label_sum['cnt'] += y_test.shape[0]
            label_sum['p_list'].extend(y_test.numpy())
            x_test = x_test.to(device)
            y_test = y_test.to(device, dtype=torch.float32)                   
            model.eval()
            outputs = model(x_test)
            prob = (outputs > threshold).float()
            test_loss = criterion(outputs, y_test)
            label_sum['pred_list'].extend(prob.cpu().numpy())
            label_sum['outputs'].extend(outputs.cpu().numpy())

            #logits.append(outputs)
            predictions.extend(prob.cpu().detach().numpy())
            test_losses.append(test_loss.item())
            labels.extend(y_test.cpu().detach().numpy())
            logits.extend(outputs.cpu().detach().numpy())
        print('labels', label_sum)

    # get metrics with NO majority vote
    acc, auc, specificity, sensitivity = get_metrics(labels, predictions,logits)
    # compute majority vote metrics
    acc_mv, auc_mv, specificity_mv, sensitivity_mv = get_majority_vote(labels, predictions)
    
    logging.info(f'FOLD {fold_c} :  test_loss_avg: {np.array(test_losses).mean()}')
    logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, specificity: {specificity}, sensitivity: {sensitivity}')

    #logging.info(f'FOLD {fold_c} :  acc_mv: {acc_mv} , auc_mv: {auc_mv}, specificity_mv: {specificity_mv}, sensitivity_mv: {sensitivity_mv}')

    test_metric=  {'acc':acc, 'auc':auc, 'sensitivity':sensitivity, 'specificity':specificity}
    test_metric_mv=  {'acc':acc_mv, 'auc':auc_mv, 'sensitivity':sensitivity_mv, 'specificity':specificity_mv}
    return test_metric, test_metric_mv

def get_transformations(is_random_crop):
    if params['model']['random_crop_size'] is None:
        data_transforms = {'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]) ]),}
        
    else:
        data_transforms = {'train': transforms.Compose([
            transforms.RandomResizedCrop(is_random_crop),
            #transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ]),'val': transforms.Compose([
            transforms.CenterCrop(is_random_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ]),} 
    return  data_transforms 

def reshape_raw_images(df, M, N):
    # Reshape the data appropriately
    logging.info('Using Raw Images')
    data = df['img'].iloc[0].view(1, M * N)
    for i in tqdm(range(1, df['img'].shape[0])):
        data = torch.cat([data, df['img'].iloc[i].view(1, M * N)])
    data = pd.DataFrame(data.numpy())
    return data

def get_scattering_features(catalog, J):
    logging.info('Importing Scattering Features')
    with open(os.path.join(catalog['data_root'], catalog[f'03_feature_scatt_{J}']), 'rb') as handle:
        scatter_dict = pickle.load(handle)
        df_scattering = scatter_dict['df']
        scattering_params = {'J':scatter_dict['J'],
                            'M':scatter_dict['M'],
                            'N':scatter_dict['N']}
    df_scattering = df_scattering.drop(columns=['id', 'class'])
    logging.info('Done Importing Scattering Features')
    return df_scattering, scattering_params

def train_predict(catalog, params):
    # panda dataframe containing flatten images - this will be use to compute eigenvectors
    df = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    if params['model']['is_raw_data']:
        data = reshape_raw_images(df, params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N'] )
    else:
        data, scattering_params = get_scattering_features(catalog, params['scattering']['J'])
    
    df = pd.concat([df, data], axis=1)
    df = df.drop(['img','fat'], axis=1)
    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c = 1
    df_pid = df['id']
    df_y = df['class']
    test_metrics = {}  
    test_metrics_mv = {}     

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Cross-validation Started')
    
    for train_index, test_index in group_kfold_test.split(df, df_y, df_pid):
        
        random.seed(seed)
        random.shuffle(train_index)
        X_test = df.iloc[test_index]
        # split training set in subtrain and validation set
        train_id, val_id = get_train_test_patients_id(df_pid.iloc[train_index], train_sz=params['model']['train_pct'], seed=seed)
        subtrain_data = df[df['id'].isin(train_id)].reset_index(drop=True).sample(frac=1,random_state =seed)
        val_data = df[df['id'].isin(val_id)].reset_index(drop=True)
        subtrain_data_flatten= subtrain_data.drop(['id','class'], axis=1)
        data_transforms = get_transformations(params['model']['random_crop_size'])

        # create datasets
        dataset_train = CldIvadoEntireDataset(subtrain_data.drop('id', axis=1), label_coln= 'class')
        dataset_val   = CldIvadoEntireDataset(val_data.drop('id', axis=1), label_coln= 'class')
        dataset_test  = CldIvadoEntireDataset(X_test.drop('id', axis=1), label_coln= 'class')

        # create  dataloaders
        dataloaders = {'train': torch.utils.data.DataLoader(dataset_train, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False, pin_memory=True),
                       'val': torch.utils.data.DataLoader(dataset_val, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False,  pin_memory=True),
                        'test': torch.utils.data.DataLoader(dataset_test, 
                                                          batch_size=params['model']['batch_size'], 
                                                          shuffle=False, pin_memory=True)}

        # If there is no transformations
        if params['model']['random_crop_size'] is None:
            # pca is used for dimensionality reduction
            logging.info(f'FOLD {fold_c}: Apply PCA on train data points')
            pca = PCA(n_components = params['pca']['n_components'], random_state = seed)          
            pca.fit(subtrain_data_flatten)
        else:
            pca = compute_pca_tranformed_images(dataloaders['train'], params['pca']['n_components'], seed)
       

        model, criterion, optimizer, scheduler = define_model(device, params['model'], 
                                                                num_components= params['pca']['n_components'],
                                                                pca= pca, is_scattering = not params['model']['is_raw_data'],
                                                                J = params['scattering']['J'],
                                                                random_crop_size= params['model']['random_crop_size'])
        
        logging.info(f'FOLD {fold_c}: model train started')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Cross-validation Started')
              
        logging.info(f'FOLD {fold_c}: model train started')
        # start training
        dataset_sizes = {'train': len(subtrain_data), 'val':  len(val_data)}
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=params['model']['epoch'],
                            patience=params['model']['patience'], threshold = params['model']['threshold'])
        logging.info(f'FOLD {fold_c}: model train done')

        # model evaluation
        test_metric, test_metric_mv = evaluate_model(model, dataloaders['test'], criterion, device, 
                                                    fold_c, threshold = params['model']['threshold'])
        test_metrics[fold_c] =  test_metric
        test_metrics_mv[fold_c] =  test_metric_mv     
        fold_c +=1 

    log_test_metrics(test_metrics, test_metrics_mv, params)
       
        
if __name__ =="__main__":
    catalog, params = get_context()
    #train_predict(catalog, params)
    #Step 3: Define a random search for these parameters, for hyperparameter tuning
    print('hey')
    random_number_generator = np.random.RandomState(0) 
    param_grid = {'lr': uniform(loc=0.000001, scale=0.0001),
                    'pca': randint(low=5, high= 500),
                    'optimizer': ['sgd', 'adam'],
                    'batch_size': [32, 64 ],
                    'epoch': randint(low=20, high= 40)}
                    #'random_crop': [224, 448],}
    param_list = list(ParameterSampler(param_grid, n_iter=params['model']['search_iter'], 
                                    random_state=params['cross_val']['seed']))
    
    # Perform hyperparameter search
    for param_dict in param_list:
        print(f"Hyperparams: num pca_comp = {param_dict['pca']}, optimizer= {param_dict['optimizer']}, lr= {round(param_dict['lr'],5)}\
                batch_size: {param_dict['batch_size']}, epoch: {param_dict['epoch']}") #random_crop: {param_dict['random_crop']}")
        params['pca']['n_components'] = param_dict['pca']
        params['model']['batch_size'] = param_dict['batch_size']
        params['model']['epoch'] = param_dict['epoch']
        params['model']['optimizer'] = param_dict['optimizer']
        params['model']['lr'] = round(param_dict['lr'], 5)
        #params['model']['random_crop_size'] =  param_dict['random_crop']
        train_predict(catalog, params) 

