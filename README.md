Classification of B-Mode Chronic Liver Disease with Deep Learning.
==============================

Methodology
------------
Split data (55 patients) into training and test sets:

- 6 patients in the test set (10%)
- 49 patients in the training set (90%)

Cross-validation on the training set is performed for hyper-parameters tuning.

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
    │   └── 03_processed   <- The final, canonical data sets for modeling.
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
