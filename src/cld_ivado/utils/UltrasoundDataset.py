import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# FROM: https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''


class UltrasoundDataset(Dataset):

    def __init__(self, data: pd.DataFrame):
        # Initialize data, download, etc.
        # read with numpy or pandas
        assert 'class' in data, f'Column class not present in dataframe.'
        assert 'img' in data, f'Column img not present in dataframe.'
    
        self.x_data = data['img'] 
        self.y_data = data['class']

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.x_data)
