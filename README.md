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
Code is provided to populated data/02_interim and data/03_features, but you must create them first.

**Mila Cluster**

It is important to note that the repository must be cloned in `/home/mila/<first_letter>/<username>/`.

According to the [Mila cluster documentation](https://mila.docs.server.mila.quebec/cluster/mila-cluster/index.html), processed data should be in the `/network/tmp1/<username>/` folder. 

To create the two folders, enter the following:
```
mkdir /network/tmp1/<username>/chronic_liver_data/02_interim/
mkdir /network/tmp1/<username>/chronic_liver_data/03_features/
```

Next, symbolic links can be created by entering the following:
```
ln -s /network/tmp1/<username>/chronic_liver_data/02_interim/ /home/mila/<first_letter>/<username>/Chronic-Liver-Classification/data/02_interim/
ln -s /network/tmp1/<username>/chronic_liver_data/03_features/ /home/mila/<first_letter>/<username>/Chronic-Liver-Classification/data/03_features/
```

**Local**

In the repository, to create the two folders, enter the following:
```
mkdir /Chronic-Liver-Classification/data/02_interim/
mkdir /Chronic-Liver-Classification/data/03_features/
```

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
