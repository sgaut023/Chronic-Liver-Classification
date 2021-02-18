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

def get_scattering_features(catalog, J, order,  M=434, N=636):
    logging.info('Importing Scattering Features')
    with open(os.path.join(catalog['data_root'], catalog[f'03_feature_scatt_{J}']), 'rb') as handle:
        scatter_dict = pickle.load(handle)
        df_scattering = scatter_dict['df']
        scattering_params = {'J':scatter_dict['J'],
                            'M':scatter_dict['M'],
                            'N':scatter_dict['N']}
    logging.info('Done Importing Scattering Features')
    df_scattering = df_scattering.drop(columns=['id', 'class'])
    if order > 0: 
        num_channels, M_prime, N_prime = get_coefficients_dimension(J, M, N)
        index_limit = 1 + 8*J
        size = df_scattering.shape[0]
        # get first order scattering
        if  order ==1:
             df_scattering = pd.DataFrame(torch.tensor(df_scattering.values).view(size, num_channels, M_prime, N_prime)[:,1:index_limit,:,:].view(size,-1).numpy())
        # get second order scattering
        elif order ==2:
             df_scattering = pd.DataFrame(torch.tensor(df_scattering.values).view(size, num_channels, M_prime, N_prime)[:,index_limit:,:,:].view(size,-1).numpy())
    return df_scattering

def get_coefficients_dimension(J, M, N):
    num_channels = 1 + 8*J + 8*8*J*(J - 1) // 2
    M_prime =  M/(2**J)
    N_prime =  N/(2**J)
    return int(num_channels), int(M_prime), int(N_prime)


def flatten_scattering(data, J, M=434, N=636): 
    num_channels, M_prime, N_prime = get_coefficients_dimension(J, M, N)

    if isinstance(data, pd.DataFrame):
        size = len(data)
        data = torch.from_numpy(data.values)
    else:
        size= data.shape[0]
    #num_channels= 960
    data = data.view(size,num_channels,M_prime, N_prime)
    data = data.reshape(-1, num_channels)
    return data, size

def reshape_scattering(x, J, M=434, N=636 ):
    num_channels, M_prime, N_prime = get_coefficients_dimension(J, M, N)
    #x = x.view(x.shape[0], 640, int(M_prime), int(N_prime) )
    #num_channels = 960
    x = x.view(x.shape[0], num_channels, M_prime, N_prime) 
    return x