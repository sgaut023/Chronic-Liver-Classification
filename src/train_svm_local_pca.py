import sys
import mlflow
import warnings
import numpy as np
import torch
import os
import pickle
from tqdm import tqdm
import random
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics, get_majority_vote,log_test_metrics

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)


def train_and_evaluate_model(parameters, X_train, X_test, y_train, y_test, fold_c):

    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters['param_grid'], verbose=parameters['verbose'], n_jobs=-1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)  # with the best found params
    print(f'predictions: {predictions}' )
    # logits = clf.predict_proba(X_valid) 

    acc, auc, specificity, sensitivity = get_metrics(y_test, predictions, predictions)
    # compute majority vote metrics
    acc_mv, auc_mv, specificity_mv, sensitivity_mv = get_majority_vote(y_test, predictions)
    
    logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, specificity: {specificity}, sensitivity: {sensitivity}')
    logging.info(f'FOLD {fold_c} :  acc_mv: {acc_mv} , auc_mv: {auc_mv}, specificity_mv: {specificity_mv}, sensitivity_mv: {sensitivity_mv}')

    test_metric = {'acc': acc, 'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}
    test_metric_mv = {'acc': acc_mv, 'auc': auc_mv, 'sensitivity': sensitivity_mv, 'specificity': specificity_mv}
    return test_metric, test_metric_mv


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

def reshape_scattering(subtrain_data_flatten, J):

    size = len(subtrain_data_flatten)
    J = params['scattering']['J']
    if J == 2:
        subtrain_data_flatten = torch.from_numpy(subtrain_data_flatten.values).view(size,108,159,81)
        subtrain_data_flatten = subtrain_data_flatten.reshape(-1,81)

    elif J == 3:
        subtrain_data_flatten = torch.from_numpy(subtrain_data_flatten.values).view(size,54,79,217)
        subtrain_data_flatten = subtrain_data_flatten.reshape(-1,217)

    elif J == 4:
        subtrain_data_flatten = torch.from_numpy(subtrain_data_flatten.values).view(size,27,39,417)
        subtrain_data_flatten = subtrain_data_flatten.reshape(-1,417)

    elif J == 5:
        subtrain_data_flatten = torch.from_numpy(subtrain_data_flatten.values).view(size,13,19,681)
        subtrain_data_flatten = subtrain_data_flatten.reshape(-1,681)

    elif J == 6:
        subtrain_data_flatten = torch.from_numpy(subtrain_data_flatten.values).view(size,6,9,1009)
        subtrain_data_flatten = subtrain_data_flatten.reshape(-1,1009)

    else:
        raise NotImplemented(f"J {self.J} parameter for scattering not implemented")
    
    return subtrain_data_flatten, size

def train_predict(catalog, params):
    df = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    J= params['scattering']['J']
    if params['model']['is_raw_data']:
        data = reshape_raw_images(df, params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N'] )
    else:
        data, scattering_params = get_scattering_features(catalog, J)
        #params['scattering']['J'] = scattering_params['J']

    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c = 1
    df_pid = df['id']
    df_y = df['class']
    test_metrics = {}  
    test_metrics_mv = {}     

    logging.info('Cross-validation Started')
    for train_index, test_index in group_kfold_test.split(df, df_y, df_pid):
        random.seed(seed)
        random.shuffle(train_index)
        X_train, X_test = data.iloc [train_index], data.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]

        X_train, size_train = reshape_scattering(X_train, J)
        X_test, size_test = reshape_scattering(X_test, J)


        # pca is used for dimensionality reduction
        logging.info(f'FOLD {fold_c}: Apply PCA on train data points')
        pca = PCA(n_components = params['pca']['n_components'], random_state = seed)          
        X_train = torch.from_numpy(pca.fit_transform(X_train)).reshape(size_train, -1)
        X_test = torch.from_numpy(pca.transform(X_test)).reshape(size_test, -1)

        if params['pca']['standardize']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        logging.info(f'FOLD {fold_c}: model train started')
        test_metric, test_metric_mv = train_and_evaluate_model(params['model'], X_train, X_test, y_train, y_test, fold_c)
        logging.info(f'FOLD {fold_c}: model train done')
        test_metrics[fold_c] =  test_metric
        test_metrics_mv[fold_c] =  test_metric_mv     
        fold_c += 1 
    log_test_metrics(test_metrics, test_metrics_mv, params)
       
        
if __name__ =="__main__":
    catalog, params = get_context()
    train_predict(catalog, params)
    # params['scattering']['J'] = 6
    # print(f"J: {params['scattering']['J']}")
    # for pca_vals in [5, 10, 15, 25,30,50,75, 100, 125, 150, 175, 200]:
    #     print(f'PCA Number of Components: {pca_vals}')
    #     params['pca']['n_components'] = pca_vals
    #     train_predict(catalog, params) 
    
    # params['scattering']['J'] = 5
    # print(f"J: {params['scattering']['J']}")
    # for pca_vals in [5, 10, 15, 25,30,50,75, 100, 125, 150, 175, 200]:
    #     print(f'PCA Number of Components: {pca_vals}')
    #     params['pca']['n_components'] = pca_vals
    #     train_predict(catalog, params) 
    
    # params['scattering']['J'] = 4
    # print(f"J: {params['scattering']['J']}")
    # for pca_vals in [5, 10, 15, 25,30,50,75, 100, 125, 150, 175, 200]:
    #     print(f'PCA Number of Components: {pca_vals}')
    #     params['pca']['n_components'] = pca_vals
    #     train_predict(catalog, params) 
    
    # params['scattering']['J'] = 3
    # print(f"J: {params['scattering']['J']}")
    # for pca_vals in [5, 10, 15, 25,30,50,75, 100, 125, 150, 175, 200]:
    #     print(f'PCA Number of Components: {pca_vals}')
    #     params['pca']['n_components'] = pca_vals
    #     train_predict(catalog, params) 
