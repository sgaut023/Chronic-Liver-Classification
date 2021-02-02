import sys
import warnings
import argparse
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint
import numpy as np
import os
import random
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from cld_ivado.utils.context import get_context
from cld_ivado.utils.reshape_features import flatten_scattering, reshape_scattering
from cld_ivado.utils.reshape_features import reshape_raw_images
from cld_ivado.utils.reshape_features import get_scattering_features
from cld_ivado.utils.compute_metrics import get_metrics
from cld_ivado.utils.compute_metrics import get_average_per_patient
from cld_ivado.utils.compute_metrics import log_test_experiments
from cld_ivado.utils.split import get_train_test_patients_id
from cld_ivado.utils.neural_networks_utils import train_model 
from cld_ivado.utils.neural_networks_utils import evaluate_model_metrics, get_dataloaders
from cld_ivado.utils.neural_networks_utils import get_all_transformations
from cld_ivado.utils.neural_networks_utils import get_normalize_transformations
from cld_ivado.utils.neural_networks_utils import create_local_scattering_layers
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing
from cld_ivado.dataset.dl_dataset import CldIvadoDataset, CldIvadoEntireDataset

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_components, pca, random_crop_size=None, is_scattering=False, J=2, global_pca = True):
        super(LogisticRegression, self).__init__()            
        if global_pca :
            shape = pca.components_[0:num_components].shape[1]
            self.pca_layer = torch.nn.Linear(shape, num_components)
            self.pca_layer.weight.data = torch.from_numpy(pca.components_[0:num_components])
            self.linear = torch.nn.Linear(num_components, 1)
        else: 
            self.linear, self.pca_layer = create_local_scattering_layers(J, num_components)
            weights = torch.from_numpy(pca.components_[0:num_components]).unsqueeze(-1).unsqueeze(-1)
            self.pca_layer.weight.data = weights.to(dtype=torch.float)

        self.is_scattering = is_scattering
        self.J = J
        self.global_pca = global_pca
        
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
            if self.global_pca:
                pca_components = self.pca_layer(x.view(x.shape[0],-1))
            else:
                x = reshape_scattering(x, self.J)
                pca_components = self.pca_layer(x)
                # flatten coefficients
                pca_components = pca_components.view(pca_components.shape[0],-1)

        y_pred = F.sigmoid(self.linear(pca_components))
        return y_pred.view(-1)


def define_model(device, params, num_components, pca, is_scattering = False, J=2, random_crop_size = None,
                global_pca = True):
    
    model_ft = LogisticRegression(num_components = num_components, pca = pca,
                                    is_scattering = is_scattering , J=J ,
                                    random_crop_size = random_crop_size, 
                                    global_pca = global_pca  )
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


def evaluate_model(model, test_loader, criterion, device, fold_c):
    # from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#5017
    test_losses = []
    probs = []
    labels = []
    label_sum = {'positive': 0,'negative': 0, 'cnt': 0,'p_list': [], 'outputs':[]}
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
            test_loss = criterion(outputs, y_test)
            label_sum['outputs'].extend(outputs.cpu().numpy())

            #logits.append(outputs)
            probs.extend(outputs.cpu().detach().numpy())
            test_losses.append(test_loss.item())
            labels.extend(y_test.cpu().detach().numpy())
        print('labels', label_sum)

    # get metrics with NO majority vote
    acc, auc, specificity, sensitivity = get_metrics(labels, probs)
    # compute majority vote metrics
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

def compute_pca_tranformed_images(dataloader, n_components, seed):

    images = []
    with torch.no_grad():
        for x_test, _ in dataloader:
            images.extend(x_test.numpy())
    pca = PCA(n_components =  n_components, random_state = seed)      
    images = np.array(images)
    images = images.reshape(images.shape[0],images.shape[2] * images.shape[3])   
    pca.fit(images)
    return pca

def train_predict(catalog, params):
    # panda dataframe containing flatten images - this will be use to compute eigenvectors
    df = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    if params['model']['is_raw_data']:
        if params['pca']['global'] is False:
            raise NotImplemented(f"Local PCA not implemented for raw images")
        data = reshape_raw_images(df, params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N'] )
         # using raw images and not scattering
        params['scattering']['J'] = None
        params['scattering']['max_order'] = None
    else:
        J = params['scattering']['J']
        data, scattering_params = get_scattering_features(catalog, params['scattering']['J'])
    
    df = pd.concat([df, data], axis=1)
    is_rgb = params['model']['is_rgb']
    # panda dataframe with path to images
    dataset = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    dataset = create_dataframe_preproccessing(dataset)
    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c = 1
    df_pid = df['id']
    df_y = df['class']
    df_fat = df['fat']

    df = df.drop(['img','fat'], axis=1)
    test_metrics = {}  
    test_metrics_avg = {}   

    labels_all, probs_all, fat_percentage  = [], [], []  # for mlflow
    patient_ids, avg_prob, label_per_patients_all  = [], [], []# for mlflow,  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Cross-validation Started')
    
    for train_index, test_index in group_kfold_test.split(df, df_y, df_pid):
        random.seed(seed)
        random.shuffle(train_index)
        fat_percentage.extend(df_fat[test_index])
        patient_ids.extend(df_pid[test_index])
 
        # If there is no transformations
        if params['model']['transform'] is False:
            data_transforms = get_normalize_transformations()
        else:
            data_transforms = get_all_transformations(params['model']['random_crop_size'], is_rgb)

        # Raw images 
        if params['model']['is_raw_data']:
            X_train, X_test, y_test = dataset.iloc[train_index], dataset.iloc[test_index], df_y.iloc[test_index]
            # split training set in subtrain and validation set
            train_id, val_id = get_train_test_patients_id(df_pid.iloc[train_index], train_sz=params['model']['train_pct'], seed=seed)
            subtrain_data = dataset[dataset['id'].isin(train_id)].reset_index(drop=True).sample(frac=1,random_state =seed)
            val_data = dataset[dataset['id'].isin(val_id)].reset_index(drop=True)
            subtrain_data_flatten = df[df['id'].isin(train_id)].reset_index(drop=True).sample(frac=1,random_state =seed)
            subtrain_data_flatten= subtrain_data_flatten.drop(['id','class'], axis=1)
            # create datasets
            dataset_train = CldIvadoDataset(subtrain_data, catalog['data_root'], 'labels', 'fname',
                                                data_transforms['train'], is_rgb = False)
            dataset_val =   CldIvadoDataset(val_data, catalog['data_root'], 'labels', 'fname',  
                                                data_transforms['val'], is_rgb = False)
            dataset_test = CldIvadoDataset(X_test, catalog['data_root'], 'labels', 'fname', 
                                               data_transforms['val'], is_rgb = False)
        # Scattering Features
        else:
            X_test = df.iloc[test_index]
            y_test = df_y[test_index]
            # split training set in subtrain and validation set
            train_id, val_id = get_train_test_patients_id(df_pid.iloc[train_index], train_sz=params['model']['train_pct'], seed=seed)
            subtrain_data = df[df['id'].isin(train_id)].reset_index(drop=True).sample(frac=1,random_state =seed)
            val_data = df[df['id'].isin(val_id)].reset_index(drop=True)
            subtrain_data_flatten= subtrain_data.drop(['id','class'], axis=1)
            dataset_train = CldIvadoEntireDataset(subtrain_data.drop('id', axis=1), label_coln= 'class')
            dataset_val   = CldIvadoEntireDataset(val_data.drop('id', axis=1), label_coln= 'class')
            dataset_test  = CldIvadoEntireDataset(X_test.drop('id', axis=1), label_coln= 'class')
            
            if params['pca']['global'] is False:
                subtrain_data_flatten, size_train = flatten_scattering(subtrain_data_flatten, J)

        # create  dataloaders
        dataloaders = get_dataloaders(dataset_train, dataset_val, dataset_test, params['model']['batch_size'])
    
        # If there is no transformations
        if params['model']['transform'] is False:
            logging.info(f'FOLD {fold_c}: Apply PCA on train data points')
            pca = PCA(n_components = params['pca']['n_components'], random_state = seed)          
            pca.fit(subtrain_data_flatten)
        else:
            logging.info(f'FOLD {fold_c}: Apply PCA on train data points')
            pca = compute_pca_tranformed_images(dataloaders['train'], params['pca']['n_components'], seed) 

        model, criterion, optimizer, scheduler = define_model(device, params['model'], 
                                                                num_components= params['pca']['n_components'],
                                                                pca= pca, is_scattering = not params['model']['is_raw_data'],
                                                                J = J,
                                                                random_crop_size= params['model']['random_crop_size'],
                                                                global_pca= params['pca']['global'])
        
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
        test_metric, test_metric_avg, probs, label_per_patient, average_prob = evaluate_model(model, dataloaders['test'], criterion, device, 
                                                                                                fold_c)
        labels_all.extend(y_test)
        probs_all.extend(probs)
        avg_prob.extend(average_prob)
        label_per_patients_all.extend(label_per_patient)
        logging.info(f'FOLD {fold_c}: model train done')
        
        test_metrics[fold_c] =  test_metric
        test_metrics_avg[fold_c] =  test_metric_avg   
        fold_c +=1 

    all_predictions = {'labels': labels_all, 'probabilities': probs_all, 
                        'Fat_percentage': fat_percentage, 'Patient ID': patient_ids}

    df_all_predictions= pd.DataFrame(data= all_predictions)
    pred_values = {'df_all_predictions':  df_all_predictions, 
                    'average_prob': avg_prob, 
                    'label_per_patient': label_per_patients_all}

    log_test_experiments(test_metrics, test_metrics_avg, params = params, pred_values = pred_values)
       
        
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='parameters_logistic.yml',
                        help="YML Parameter File Name")
    args = parser.parse_args()
    catalog, params = get_context(args.param_file)
    train_predict(catalog, params)
    #Step 3: Define a random search for these parameters, for hyperparameter tuning
    #random_number_generator = np.random.RandomState(0) 
    # param_grid = {'lr': uniform(loc=0.00001, scale=0.001),
    #                 'pca': randint(low=5, high= 200),
    #                 'optimizer': ['sgd', 'adam'],
    #                 'random_crop': [224, 448],}
    # param_list = list(ParameterSampler(param_grid, n_iter=params['model']['search_iter'], 
    #                                 random_state=params['cross_val']['seed']))
    
    # # Perform hyperparameter search
    # for param_dict in param_list:
    #     print(f"Hyperparams: num pca_comp = {param_dict['pca']}, optimizer= {param_dict['optimizer']}, lr= {round(param_dict['lr'],5)}\
    #             random_crop: {param_dict['random_crop']}")
    #     params['pca']['n_components'] = param_dict['pca']
    #     params['model']['optimizer'] = param_dict['optimizer']
    #     params['model']['lr'] = round(param_dict['lr'], 5)
    #     params['model']['random_crop_size'] =  param_dict['random_crop']
    #     train_predict(catalog, params) 

