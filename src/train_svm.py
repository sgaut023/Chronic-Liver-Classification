import sys
import math
import numpy as np
import warnings
import torch
import os
import pickle
from tqdm import tqdm
import argparse
import random
import pandas as pd
import logging
from sklearn.model_selection import GroupKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cld_ivado.utils.context import get_context
from cld_ivado.utils.compute_metrics import get_metrics, get_average_per_patient, log_test_experiments

sys.path.append('../src')
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)


def train_and_evaluate_model(parameters, X_train, X_test, y_train, y_test, fold_c):
    """
    :param parameters: parameters to train model
    :param X_train: training data points
    :param X_test: testing data points
    :param y_train: training labels
    :param y_test: testing label
    :param fold_c: fold number
    """

    svc = svm.SVC(probability = True)
    clf = GridSearchCV(svc, parameters['param_grid'], verbose=parameters['verbose'], n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = np.array(clf.predict_proba(X_test))[:,1]
    print(f'labels: {np.array(y_test)}')
    print(f'predictions: {probs}')

    acc, auc, specificity, sensitivity = get_metrics(y_test, probs)
    acc_avg, auc_avg, specificity_avg, sensitivity_avg, average_prob = get_average_per_patient(y_test, probs)
    

    if math.isnan(auc):
        logging.info(f'FOLD {fold_c} :  acc: {acc} , specificity: {specificity}, sensitivity: {sensitivity}')
        logging.info(f'Average per patient:  acc : {acc_avg} , specificity: {specificity_avg}, sensitivity: {sensitivity_avg}')

    else:
        logging.info(f'FOLD {fold_c} :  acc: {acc} , auc: {auc}, \
                    specificity: {specificity}, sensitivity: {sensitivity}')
        logging.info(f'Average per patient:  acc : {acc_avg} , auc: {auc_avg},\
                    specificity: {specificity_avg}, sensitivity: {sensitivity_avg}')

    test_metric = {'acc': acc, 'auc': auc, 'sensitivity': sensitivity, 'specificity': specificity}
    test_metric_avg = {'acc': acc_avg, 'auc': auc_avg, 'sensitivity': sensitivity_avg, 'specificity': specificity_avg}
    return test_metric, test_metric_avg, probs, average_prob


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
    df = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
    if params['model']['is_raw_data']:
        data = reshape_raw_images(df, params['preprocess']['dimension']['M'], params['preprocess']['dimension']['N'] )
        df = df.drop(columns=['img'])
    else:
        data, scattering_params = get_scattering_features(catalog, params['scattering']['J'])

    test_n_splits = params['cross_val']['test_n_splits']
    group_kfold_test = GroupKFold(n_splits=test_n_splits)
    seed = params['cross_val']['seed']
    fold_c = 1
    df_pid = df['id']
    df_y = df['class']
    df_fat = df['fat']

    # save metrics and probability
    test_metrics = {}  
    test_metrics_avg = {}     
    labels_all, probs_all, fat_percentage  = [], [], []  # for mlflow
    patient_ids, avg_prob, label_per_patients_all  = [], [], []# for mlflow,

    logging.info('Cross-validation Started')
    for train_index, test_index in group_kfold_test.split(df, df_y, df_pid):
        random.seed(seed)
        random.shuffle(train_index)
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test, y_fat = df_y.iloc[train_index], df_y.iloc[test_index], df_fat[test_index]

        fat_percentage.extend(y_fat)
        patient_ids.extend(y_test)

        # pca is used for dimensionality reduction
        logging.info(f'FOLD {fold_c}: Apply PCA on train data points')
        pca = PCA(n_components = params['pca']['n_components'], random_state = seed)          
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        #standardize
        if params['pca']['standardize']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        logging.info(f'FOLD {fold_c}: model train started')
        # training and evalution
        test_metric, test_metric_avg, probs, label_per_patient, average_prob = train_and_evaluate_model(params['model'], 
                                                                        X_train, X_test,
                                                                         y_train, y_test, 
                                                                         fold_c = fold_c)
        labels_all.extend(y_test)
        probs_all.extend(probs)
        avg_prob.extend(average_prob)
        label_per_patients_all.extend(label_per_patient)
        logging.info(f'FOLD {fold_c}: model train done')
        
        test_metrics[fold_c] =  test_metric
        test_metrics_avg[fold_c] =  test_metric_avg     
        
        fold_c += 1 
    
    # log all the metrics in mlflow
    all_predictions = {'labels': labels_all, 'probabilities': probs_all, 
                        'Fat_percentage': fat_percentage, 'Patient ID': patient_ids }

    df_all_predictions= pd.DataFrame(data= values)
    pred_values = {'df_all_predictions':  df_all_predictions, 
                    'average_prob': average_prob, 
                    'label_per_patient': label_per_patient}

    log_test_experiments(test_metrics, test_metrics_avg, params = params, pred_values = pred_values)
       
        
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, default='parameters_svm.yml',
                        help="YML Parameter File Name")
    args = parser.parse_args()
    catalog, params = get_context(args.param_file)
    train_predict(catalog, params) 
    
