import numpy as np
import pandas as pd
import random
import mlflow
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#From: https://www.thetopsites.net/article/52106959.shtml
def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)


def get_metrics(labels, preds):
    '''
    Fonction that compute the accuracy, the AUC score, the specificity and the sensitivity
    based on the labels and predictions
    '''
    acc = accuracy_score(labels, preds)
    #TO DO: FIX
#     if len(np.unique(labels)) == 1 and len(np.unique(preds)) ==1:
#         auc = acc
#         specificity = acc
#         sensitivity = acc
#    else: 
    auc = roc_auc_score_FIXED(labels, preds)
    if len(np.unique(labels)) == 1 and len(np.unique(preds)) ==1 and np.unique(preds) == np.unique(labels):
        if np.unique(preds) ==1:
            specificity = np.nan
            sensitivity = 1.   
        else:
            specificity = 1.
            sensitivity = np.nan  
    else:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
    return acc, auc, specificity, sensitivity

def get_majority_vote(y_test, predictions):
    # We assume that every patient has 10 ultrasound images in sequence
    # Thus, the first 0-9 images are from the same patient
    # The second 10 images (10-19) are from another patient and so on
    majority_vote_predictions= []
    majority_vote_labels = [] 
    num_img_per_pat = 10
    for i in range(0,len(predictions),  num_img_per_pat):
        idx = np.arange(i, i+ num_img_per_pat)
        counter_pred = Counter(predictions[idx])
        majority_vote_predictions.append(int(counter_pred.most_common(1)[0][0]))
        majority_vote_labels.append(int(np.array(y_test)[i]))
    return get_metrics(majority_vote_labels, majority_vote_predictions)

def log_mlflow_metrics(acc, auc,specificity, sensitivity): 
    mlflow.log_metric('accuracy mean',np.nanmean(acc))
    mlflow.log_metric('AUC mean',np.nanmean(auc))
    mlflow.log_metric('specificity mean', np.nanmean(specificity))
    mlflow.log_metric('sensitivity mean',np.nanmean(sensitivity))
    #log variance of metrics
    mlflow.log_metric('accuracy variance',np.nanvar(acc))
    mlflow.log_metric('AUC variance', np.nanvar(auc))
    mlflow.log_metric('specificity variance', np.nanvar(specificity))
    mlflow.log_metric('sensitivity variance', np.nanvar(sensitivity))
    

def log_test_metrics(test_metrics, test_metrics_mv, test_n_splits, model_name, seed):
    '''
    Functions that log test metrics with MLFLOW
    '''

    test_acc = np.array([np.array(test_metrics[fold]['acc']) for fold in range(1, test_n_splits+1)])
    test_auc = np.array([np.array(test_metrics[fold]['auc']) for fold in range(1, test_n_splits+1)])
    test_sensitivity = np.array([np.array(test_metrics[fold]['sensitivity']) for fold in range(1, test_n_splits+1)])
    test_specificity = np.array([np.array(test_metrics[fold]['specificity']) for fold in range(1, test_n_splits+1)])
    #majority vote metrics
    test_acc_mv = np.array([np.array(test_metrics_mv[fold]['acc']) for fold in range(1, test_n_splits+1)])
    test_auc_mv = np.array([np.array(test_metrics_mv[fold]['auc']) for fold in range(1, test_n_splits+1)])
    test_sensitivity_mv = np.array([np.array(test_metrics_mv[fold]['sensitivity']) for fold in range(1, test_n_splits+1)])
    test_specificity_mv = np.array([np.array(test_metrics_mv[fold]['specificity']) for fold in range(1, test_n_splits+1)])

    #log params
    mlflow.set_experiment('experiment_per_model')
    with mlflow.start_run():   
        mlflow.log_param('Model', model_name)
        mlflow.log_param('Seed', seed)
        mlflow.log_param('Majority Vode', 'No')
        mlflow.log_param('Number of Folds', test_n_splits)
        # No majority VOTE
        log_mlflow_metrics(test_acc, test_auc,test_specificity, test_sensitivity)
        
    with mlflow.start_run():
        # Majority VOTE
        mlflow.log_param('Model', model_name)
        mlflow.log_param('Majority Vode', 'Yes')
        mlflow.log_param('Number of Folds', test_n_splits)
        log_mlflow_metrics(test_acc_mv, test_auc_mv,test_specificity_mv, test_sensitivity_mv)
        print('Experiment done')  