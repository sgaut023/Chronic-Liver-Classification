import numpy as np
import pandas as pd
import random
import torch 

def get_train_test_patients_id(ids, train_sz: float=.9, seed:int=2020):
    # Get the list of patient id's and shuffle that list 
    patient_ids = ids.unique()
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    # Create two lists of patient id's for training and testing
    train_patient_cnt = int(len(patient_ids) * train_sz)
    train_id = patient_ids[0:int(train_patient_cnt)]
    test_id = patient_ids[int(train_patient_cnt): int(len(patient_ids))]
    return train_id, test_id


def train_test_split(data: pd.DataFrame, train_sz:float=.9, seed:int=2020):
    """
    Function that will split the dataset into training and testing based
    on patient id's. A patient will not appear in both training and testing set.
    
    """
    assert 'id' in data, f'Column id not present in dataframe.'
    
    train_id, test_id = get_train_test_patients_id(data['id'],train_sz , seed)
    # Separate the features from id and class columns
    train_data = data[data['id'].isin(train_id)].reset_index(drop=True).sample(frac=1,random_state =seed)
    test_data = data[data['id'].isin(test_id)].reset_index(drop=True)
    return train_data, test_data

def train_test_split_pytorch(data:torch.Tensor, ids, labels, train_sz:float=.9, seed:int=2020):

    train_id, test_id = get_train_test_patients_id(ids,train_sz, seed )
    train_indexes = ids[ids.isin(train_id)].index
    test_indexes = ids[ids.isin(test_id)].index
    train_id, test_id = get_train_test_patients_id(ids )
    train_indexes = ids[ids.isin(train_id)].index
    test_indexes = ids[ids.isin(test_id)].index
    
    random.shuffle(train_indexes.values)
    random.shuffle(test_indexes.values)


    return data[train_indexes], labels[train_indexes], data[test_indexes], labels[test_indexes]
    

