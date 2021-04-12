import numpy as np
import pandas as pd
import mlflow
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, roc_curve
import logging
logging.basicConfig(level = logging.INFO)

def get_metrics(labels, probs, threshold = 0.5):
    """
        Fonction that compute the accuracy, the AUC score, the specificity and the sensitivity
        based on the labels and predictions
    """
    try:
        auc = roc_auc_score(labels,  probs)
        #logging.info(f'Threshold used based on roc_curve: {threshold}')
    except ValueError:
        auc = np.nan
        #logging.info(f'Default Threshold used: {threshold}')
    
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

    return get_metrics(np.array(labels_all), np.array(average_prob)), np.array(labels_all), np.array(average_prob) 

def get_metrics_from_dictionnary(test_metrics, params, average = False): 
    test_n_splits = params['cross_val']['test_n_splits']
    accs = np.array([np.array(test_metrics[fold]['acc']) for fold in range(1, test_n_splits+1)])
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
   
def plot_roc_curve(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(8,8))
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return f

def get_roc_curve_per_fold(labels, predictions, n_fold):
    size_fold = int(len(labels)/n_fold)
    figure = plt.figure()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver Operating Characteristic')
    for i in range(n_fold):
        fpr, tpr, thresh = roc_curve(np.array(labels)[i: i+size_fold], np.array(predictions)[i: i+size_fold])
        auc = roc_auc_score(np.array(labels)[i*size_fold: i*size_fold+size_fold], np.array(predictions)[i*size_fold: i*size_fold+size_fold])
        plt.plot(fpr,tpr,label=f'Fold {i+1}: AUC: {round(auc*100,2)}')
    plt.legend(loc=0)
    return figure 
def log_test_experiments(test_metrics, test_metrics_avg, params, pred_values):
    '''
        Functions that log test metrics using MLFLOW tool
        The metrics are averaged over the different folds
        :param test_metrics: dictionnary of the different metrics, keys: acc, auc, sensitivity and specificity
        :param params: parameters to log
    '''
    metrics = get_metrics_from_dictionnary(test_metrics, params)
    metrics_avg = get_metrics_from_dictionnary(test_metrics_avg, params, average = True)
    pred_overall = pd.DataFrame(data = {'labels':pred_values['label_per_patient'],
                                            'pred': pred_values['average_prob'] })
    # save predictions
    pd.DataFrame.from_dict(test_metrics).to_csv('fold_metrics.csv')
    pd.DataFrame.from_dict(test_metrics_avg).to_csv('fold_metrics_avg.csv')
    pred_values['df_all_predictions'].to_csv('all_predictions.csv')
    pred_overall.to_csv('overall_predictions.csv')

    #create the ROC curve
    figure = plot_roc_curve(pred_values['df_all_predictions']['labels'], pred_values['df_all_predictions']['probabilities'])
    figure_avg= plot_roc_curve(pred_values['label_per_patient'], pred_values['average_prob'])

    figure_folds = get_roc_curve_per_fold(pred_values['df_all_predictions']['labels'], 
                                            pred_values['df_all_predictions']['probabilities'],
                                            params['cross_val']['test_n_splits'])
    #create a ROC curce for every folds


    try:
        auc_overall = roc_auc_score(pred_values['label_per_patient'],  pred_values['average_prob']) #over the different folds
    except ValueError:
        auc_overall  = np.nan
    log_test_experiments_mlflow(metrics, metrics_avg, params, figure = [figure_avg, figure,figure_folds], auc_overall  = auc_overall )
    os.remove('all_predictions.csv')
    os.remove('overall_predictions.csv')
    os.remove('fold_metrics.csv')
    os.remove('fold_metrics_avg.csv')


def log_test_experiments_mlflow(metrics, metrics_avg, params, figure, auc_overall ):
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
        mlflow.log_artifact('overall_predictions.csv', 'predictions')
        mlflow.log_artifact('fold_metrics.csv', 'predictions')
        mlflow.log_artifact('fold_metrics_avg.csv', 'predictions')
        mlflow.log_figure(figure[0], 'roc_avg.png')
        mlflow.log_figure(figure[1], 'roc_all.png')
        mlflow.log_figure(figure[2], 'roc_fold.png')
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(metrics_avg)
        mlflow.log_metric('AVG AUC overall', auc_overall )
    print('Experiment done')  