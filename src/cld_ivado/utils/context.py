from pathlib import Path
import yaml
import os
import sys

def get_context(parameters_file):
    # Get the current project path (where you open the notebook)
    # and go up two levels to get the project path
    current_dir = Path.cwd()
    #proj_path = current_dir.parent
    proj_path = current_dir
    # make the code in src available to import in this notebook
    sys.path.append(os.path.join(proj_path, 'src'))

    # Catalog contains all the paths related to datasets
    with open(os.path.join(proj_path, 'conf/data_catalog.yml'), "r") as f:
        catalog = yaml.safe_load(f)
        
    # Params contains all of the dataset creation parameters and model parameters
    with open(os.path.join(proj_path, f'conf/{parameters_file}'), "r") as f:
        params = yaml.safe_load(f)
    return catalog, params