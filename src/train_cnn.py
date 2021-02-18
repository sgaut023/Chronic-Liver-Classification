import sys
import warnings
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint
import os
import random
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold
from cld_ivado.utils.neural_networks_utils import train_model, evaluate_model_metrics
from cld_ivado.utils.neural_networks_utils import get_all_transformations
from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import log_test_experiments
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing
from cld_ivado.utils.split import train_test_split
from cld_ivado.dataset.dl_dataset import CldIvadoDataset

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=params['kernel_size'])
        self.conv2 = nn.Conv2d(32, 32, kernel_size=params['kernel_size'])
        self.conv3 = nn.Conv2d(32, 64, kernel_size=params['kernel_size'])
        if params['is_rgb'] is False:
            #self.fc1 = nn.Linear(186624, 256)
            self.fc1 = nn.Linear(1065088, 256)
        else:
            self.fc1 = nn.Linear(276480, 256)
        
        self.fc2 = nn.Linear(256, 2)
        self.dropout = params['dropout']


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(x.shape[0],  -1)
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


def evaluate_model(model, test_loader, criterion, device, fold_c):
    # from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e#5017
    test_losses = []
    probs = []
    labels = []
    label_sum={'positive':0,'negative':0, 'cnt':0,'p_list':[], 'outputs': []}

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
            test_loss = criterion(outputs, y_test)
            label_sum['outputs'].extend(prob.cpu().numpy()[:, 1])

            test_losses.append(test_loss.item())
            labels.extend(y_test.cpu().detach().numpy())
            probs.extend(prob[:, 1].cpu().detach().numpy())
        print('labels', label_sum)
    
    logging.info(f'FOLD {fold_c} :  test_loss_avg: {np.array(test_losses).mean()}')
    return evaluate_model_metrics(labels, probs, fold_c)

def train_predict(catalog, params):
    dataset = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    dataset = create_dataframe_preproccessing(dataset)
  
    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c =1 

    df_pid = dataset['id']
    df_y = dataset['labels']
    df_fat = dataset['fat']

    test_metrics ={}  
    test_metrics_avg = {}
    labels_all, probs_all, fat_percentage = [], [], []          # for mlflow
    patient_ids, avg_prob, label_per_patients_all = [], [], []  # for mlflow,
    
    is_rgb = params['model']['is_rgb']
    data_transforms = get_all_transformations(params['model']['random_crop_size'], is_rgb)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Cross-validation Started')
    for train_index, test_index in group_kfold_test.split(dataset, df_y, df_pid):

        random.seed(seed)
        random.shuffle(train_index)
        X_train, X_test, y_test = dataset.iloc[train_index], dataset.iloc[test_index], df_y[test_index]
        fat_percentage.extend(df_fat[test_index])
        patient_ids.extend(df_pid[test_index])
        labels_all.extend(y_test)

        model, criterion, optimizer, scheduler = define_model(device, params['model'])

        # split training set in subtrain and validation set
        subtrain_data, val_data = train_test_split(X_train, train_sz=params['model']['train_pct'], seed=seed)

        dataset_train = CldIvadoDataset(subtrain_data, catalog['data_root'], 'labels', 'fname', 
                                        data_transforms['train'], is_rgb = is_rgb)

        dataset_val = CldIvadoDataset(val_data, catalog['data_root'], 'labels', 'fname',  
                                        data_transforms['val'],  is_rgb = is_rgb)

        dataset_test = CldIvadoDataset(X_test, catalog['data_root'], 'labels', 'fname', 
                                        data_transforms['val'],  is_rgb = is_rgb)

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
        model = train_model(model, criterion, optimizer, scheduler, 
                            dataloaders, device, dataset_sizes, 
                            num_epochs=params['model']['epoch'],
                            patience=params['model']['patience'])
        logging.info(f'FOLD {fold_c}: model train done')

        # model evaluation
        test_metric, test_metric_avg, probs, label_per_patient, average_prob = evaluate_model(model, 
                                                                                            dataloaders['test'], 
                                                                                            criterion, device,                                                                                         fold_c)
        probs_all.extend(probs)
        label_per_patients_all.extend(label_per_patient)
        avg_prob.extend(average_prob)
        test_metrics[fold_c] =  test_metric
        test_metrics_avg[fold_c] =  test_metric_avg    
        fold_c +=1 

    # log all the metrics in mlflow
    all_predictions = {'labels': labels_all, 'probabilities': probs_all, 
                        'Fat_percentage': fat_percentage, 'Patient ID': patient_ids}

    df_all_predictions= pd.DataFrame(data= all_predictions)
    pred_values = {'df_all_predictions':  df_all_predictions, 
                    'average_prob': avg_prob, 
                    'label_per_patient': label_per_patients_all}

    log_test_experiments(test_metrics, test_metrics_avg, params = params, pred_values = pred_values)
       
        
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='parameters_cnn.yml',
                        help="YML Parameter File Name")
    args = parser.parse_args()
    catalog, params = get_context(args.param_file)
    #train_predict(catalog, params) 
    
    #train_predict(catalog, params)
    # for n_split in [3,4,5,6,7,8,10,11]:
    #     #print(f'PCA Number of Components: {pca_vals}')
    #     params['cross_val']['test_n_splits'] = n_split
    #     train_predict(catalog, params) 


    param_grid = {'lr': uniform(loc=0.000001, scale=0.001),'dropout': uniform(loc=0.0, scale=0.5)}
    param_list = list(ParameterSampler(param_grid, n_iter=params['model']['search_iter'], 
                                    random_state=42))

                        
    #Perform hyperparameter search
    for param_dict in param_list:
        params['model']['lr'] = round(param_dict['lr'], 5)
        params['model']['dropout'] = param_dict['dropout']
        train_predict(catalog, params) 

