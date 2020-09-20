import numpy as np
import pandas as pd

def train_test_split(data: pd.DataFrame, train_sz:float=.9, seed:int=2020):
    """
    Function that will split the dataset into training and testing based
    on patient id's. A patient will not appear in both training and testing set.
    
    :param data
    :param seed 
    """
    assert 'id' in data, f'Column id not present in dataframe.'
    
    # Get the list of patient id's and shuffle that list 
    patient_ids = data['id'].unique()
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    # Create two lists of patient id's for training and testing
    train_patient_cnt = int(len(patient_ids) * train_sz)
    train_id = patient_ids[0:int(train_patient_cnt)]
    test_id = patient_ids[int(train_patient_cnt): int(len(patient_ids))]
    # Separate the features from id and class columns
    train_data = data[data['id'].isin(train_id)].reset_index(drop=True)
    test_data = data[data['id'].isin(test_id)].reset_index(drop=True)
    
    return train_data, test_data
