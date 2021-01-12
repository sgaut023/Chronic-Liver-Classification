import sys
sys.path.append('../src')
from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from pathlib import Path
import os
import logging
logging.basicConfig(level = logging.INFO)
from cld_ivado.utils.context import get_context


def export_us_mat_to_png():
    catalog, params = get_context()
    M= params['preprocess']['dimension']['M']
    N= params['preprocess']['dimension']['N']
    data= loadmat(os.path.join(catalog['data_root'],catalog['01_raw_mat']))
    df = pd.DataFrame()

    logging.info('Extracting US images from .mat')
    for patient in data['data'][0]:
        p_id = patient[0][0][0]
        p_class = patient[1][0][0]
        p_fat = patient[2][0][0]
        for i,p_image in enumerate(patient[3]):
            
            
            patient_row = pd.DataFrame({'id':p_id,
                                        'class':p_class,
                                        'fat':p_fat,
                                        'img':[torch.tensor(p_image).view(1,M,N).type(torch.float32)]})
            df = df.append(patient_row,)
            Image.fromarray(np.uint8(p_image)).save(os.path.join(catalog['data_root'],f'data/01_raw/raw_images/P{p_id}_image{i+1}.jpg'))

    df.reset_index(drop=True, inplace=True)

    #save dataset in order to use it later
    with open(os.path.join(catalog['data_root'],catalog['02_interim_pd']), 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    logging.info(f"PNG images saved {os.path.join(catalog['data_root'],'data/01_raw/raw_images/')}")
    logging.info(f"Panda dataset saved {os.path.join(catalog['data_root'],catalog['02_interim_pd'])}")

if __name__ =="__main__":
    export_us_mat_to_png()
