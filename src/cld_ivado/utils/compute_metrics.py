import numpy as np
import mlflow
import os
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, plot_roc_curve
import logging
logging.basicConfig(level = logging.INFO)

def get_metrics(labels, probs, threshold = 0.5):
    """
        Fonction that compute the accuracy, the AUC score, the specificity and the sensitivity
        based on the labels and predictions
    """
    try:
        auc = roc_auc_score(labels,  probs)
        logging.info(f'Threshold used based on roc_curve: {threshold}')
    except ValueError:
        auc = np.nan
        logging.info(f'Default Threshold used: {threshold}')
    
    preds = [int(elem) for elem in np.array(probs) > threshold]
    acc = accuracy_score(labels, preds)
    
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



def get_average_per_patient(labels, probability, num_img_per_pat =10):
    """
    We assume that every patient has 10 ultrasound images in sequence
    Thus, the first 0-9 images are from the same patient
    The second 10 images (10-19) are from another patient and so on
    :param labels: labels
    :param probability: probability that the US images is class 1
    :param num_img_per_pat: number of images associated with one patient
    :return: acc, auc, specificity, sensitivity
    """ 
    average_prob = []
    labels_all = [] 

    for i in range(0,len(probability),  num_img_per_pat):
        idx = np.arange(i, i + num_img_per_pat)
        average_prob.append(np.array(probability)[idx].mean())
        labels_all.append(int(np.array(labels)[i]))

    return get_metrics(np.array(labels_all)) , np.array(labels_all), np.array(average_prob) 

def get_metrics_from_dictionnary(test_metrics, params, average = False): 
    test_n_splits = params['cross_val']['test_n_splits']
    accs = array([np.array(test_metrics[fold]['acc']) for fold in range(1, test_n_splits+1)])
    aucs = np.array([np.array(test_metrics[fold]['auc']) for fold in range(1, test_n_splits+1)])
    sensitivity_all = np.array([np.array(test_metrics[fold]['sensitivity']) for fold in range(1, test_n_splits+1)])
    specificity_all = np.array([np.array(test_metrics[fold]['specificity']) for fold in range(1, test_n_splits+1)])
        
    metrics = {'ACC': np.nanmean(accs), 'AUC': np.nanmean( aucs), 
                'Sensitivity': np.nanmean(sensitivity_all), 
                'Specificity': np.nanmean(specificity_all),
                'ACC var': np.nanvar(accs),'AUC var': np.nanvar(aucs),
                'Sensitivity var': np.nanvar(sensitivity_all), 
                'Specificity var': np.nanvar(specificity_all)}

    if average: 
        metrics = {'AVG- ' + str(key): val for key, val in metrics.items()}
    return metrics
   
def get_roc_curve(labels, probs):
    return plot_roc_curve(labels, probs)

def log_test_experiments(test_metrics, test_metrics_avg, params, pred_values):
    '''
        Functions that log test metrics using MLFLOW tool
        The metrics are averaged over the different folds
        :param test_metrics: dictionnary of the different metrics, keys: acc, auc, sensitivity and specificity
        :param params: parameters to log
    '''
    metrics = get_metrics_from_dictionnary(test_metrics, params)
    metrics_avg = get_metrics_from_dictionnary(test_metrics_avg, params, average = True)
    pred_values['df_all_predictions'].to_csv('all_predictions.csv')
    figure = get_roc_curve(pred_values['label_per_patient'], pred_values['average_prob'])
    log_test_experiments(metrics, metrics_avg, params, figure)
    os.remove('all_predictions.csv')


def log_test_experiments(metrics, metrics_avg, params, figure):
    '''
    Functions that log test metrics using MLFLOW tool
    The metrics are averaged over the different folds
    :param metrics: accuracy, auc score, sensitivity and specificity
    :param params: parameters to log
    '''
    logging.info(f'Metrics per images: {metrics}, Metrics per patient: {metrics_avg}')
    # log metrics and params MLFLOW
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    with mlflow.start_run():   
        mlflow.log_params(params['model'])
        mlflow.log_params(params['cross_val'])
        mlflow.log_params(params['scattering'])
        mlflow.log_params(params['pca'])
        mlflow.log_params(params['preprocess']['dimension'])
        mlflow.log_artifact('all_predictions.csv', 'predictions')
        
        mlflow.log_figure(figure)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(metrics_avg)
    print('Experiment done')  