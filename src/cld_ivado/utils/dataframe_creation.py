import numpy as np
import pandas as pd
import random

def create_dataframe_preproccessing(num_patient = 55):
    '''
    Create panda dataframe to be fed to fastai ImageDataLoaders or keras ImageDataGenerator
    ImageDataLoaders and ImageDataGenerator required the same pada dataframe structure
    '''
    dataset = pd.read_pickle('../data/02_interim/bmodes_steatosis_assessment_IJCARS.pickle')
    # sequence of 10 to represent the suffix in the image file name
    # panda dataframe that contains the path to the jpg image and its corresponding label
    list_of_seq10 = [np.arange(1,11).astype('str')] * num_patient
    list_of_seq10 = [t for tt in list_of_seq10 for t in tt]

    dataset['fname'] = [f"data/01_raw/raw_images/P{d_id}_image{d_pid}.jpg" for d_id,d_pid in list(zip(dataset['id'],list_of_seq10))]
    dataset.rename(columns={'class':'labels'},inplace=True)
    dataset.drop(columns=['img'], inplace=True)
    return dataset