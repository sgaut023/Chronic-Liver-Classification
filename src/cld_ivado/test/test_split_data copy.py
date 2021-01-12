import numpy as np
import pandas as pd
import sys
import unittest
import os
from pathlib import Path 
sys.path.append(str(Path.cwd() / 'src'))
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
    

    
if __name__ == "__main__":
    unittest.main()
    print("All test cases passed")