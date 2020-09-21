Classification of B-Mode Chronic Liver Disease with Deep Learning
==============================

Methodology
------------
Split data (55 patients) into training and test sets:

- 6 patients in the test set (10%)
- 49 patients in the training set (90%)

Cross-validation on the training set is performed for hyper-parameters tuning.

Get Setup
------------

Prerequisites
- Anaconda/Miniconda

Start by cloning the repository

To create the `ultra` conda environment, enter the following in the command prompt: 
```
conda env create -f environment.yml
```
To active the `ultra` conda environment, enter the following: 
```
conda activate ultra
```

Two directories are ignored by the .gitignore file to avoid uploading large files to github. However, the raw dataset is inclued in data/01_raw.
Code is provided to populated data/02_interim and data/03_features, but you must create

**Mila Cluster**

**Local**


Experiments
------------


Preliminary Results
------------



Project Organization
------------

    ├── LICENSE            <- None for now
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── 01_raw         <- The original, immutable data dump.
    │   ├── 02_interim     <- Intermediate data that has been transformed.
    │   └── 03_features    <- The final, canonical data sets for modeling.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── environment.yml    <- The conda environment file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.
        └── __init__.py    <- Makes src a Python module

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
