import torch
import os
import pandas as pd
import pickle
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

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

def flatten_scattering(data, J):
    
    num_channels = 1 + 8*J + 8*8*J*(J - 1) // 2
    if isinstance(data, pd.DataFrame):
        size = len(data)
        data = torch.from_numpy(data.values)
    else:
        size= data.shape[0]
    if J == 2:
        data = data.view(size,108,159,81)
        data = data.reshape(-1,81)

    elif J == 3:
        data = data.view(size,54,79,217)
        data = data.reshape(-1,217)

    elif J == 4:
        data = data.view(size,27,39,417)
        data = data.reshape(-1,417)

    elif J == 5:
        data = data.view(size,13,19,681)
        data = data.reshape(-1,681)

    elif J == 6:
        data = data.view(size,6,9,1009)
        data = data.reshape(-1,1009)

    else:
        raise NotImplemented(f"J {self.J} parameter for scattering not implemented")
    
    return data, size

def reshape_scattering(x, J):
    if J  == 2:
        x= x.view(x.shape[0], 81, 108, 159)
    elif J==3:
        x = x.view(x.shape[0], 217, 54, 79)
    elif J==4:
        x = x.view(x.shape[0], 417, 27, 39)
    elif J==5:
        x = x.view(x.shape[0], 681, 13, 19)
    elif J==6:
        x = x.view(x.shape[0], 1009, 6, 9)
    else:
        raise NotImplemented(f"J {J} parameter for scattering not implemented")
    
    return x