import numpy as np
import pandas as pd
import sys
import random
import unittest
import os
from pathlib import Path 
sys.path.append(str(Path.cwd() / 'src'))
from sklearn.model_selection import GroupKFold

from cld_ivado.utils.split import train_test_split
from cld_ivado.utils.context import get_context
from cld_ivado.utils.dataframe_creation import create_dataframe_preproccessing


class SplitTest(unittest.TestCase):
    def test_intersection(self):
        dummy = pd.DataFrame({'other_col': np.arange(10,30,2), 
                            'id':[1,1,2,2,3,3,4,4,5,5]})
        
        train, test = train_test_split(dummy)
        
        set_train = set(train['id'])
        set_test = set(test['id'])
        
        self.assertTrue(len(set_train & set_test) == 0)
    
    def test_mat_data(self):
        catalog, params = get_context()
        dataset = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
        dataset = create_dataframe_preproccessing(dataset)
        
        train_data, test_data = train_test_split(dataset, train_sz=.9)
        
        set_train = set(train_data['id'])
        set_test = set(test_data['id'])
        
        self.assertTrue(len(set_train & set_test) == 0)
    
    def test_K_group_fold(self):
        '''
        Test the K group fold and make sure the data is shuffle
        '''
        catalog, params = get_context()
        dataset = pd.read_pickle(os.path.join(catalog['data_root'], catalog['02_interim_pd']))
        dataset = create_dataframe_preproccessing(dataset)
        df_pid = dataset['id']
        df_y = dataset['labels']
    
        test_n_splits = params['cross_val']['test_n_splits']
        group_kfold_test = GroupKFold(n_splits=test_n_splits)
        seed = 11

        for train_index, test_index in group_kfold_test.split(dataset, df_y, df_pid):

            random.seed(seed)
            random.shuffle(train_index)
            X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
            set_train = set(X_train['id'].values)
            set_test = set(X_test['id'].values)
            #set_train & set_test) == 0)

            # split training set in subtrain and validation set
            subtrain_data, val_data = train_test_split(X_train, train_sz=0.9, seed=seed)
            set_subtrain = set(subtrain_data['id'].values)
            set_val = set(val_data['id'].values)
            self.assertTrue(len(set_subtrain & set_val) == 0)
    

    
if __name__ == "__main__":
    unittest.main()
    print("All test cases passed")