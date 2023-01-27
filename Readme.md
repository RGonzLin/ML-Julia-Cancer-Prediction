# Machine Learning with Julia & cancer type prediction

This repository contains a set of useful functions for Machine Learning implemented using Julia, and libraries like Flux and Scikit-learn.  
Moreover, these functions are used to predict the type of tumor samples into one of five categories based on RNA-seq data; providing a straightforward example of its usage, and additionally serving to introduce core Machine Learning concepts.  

## Key features
* Data normalization
* One-hot encoding
* Artificial Neural Networks with early stopping
* Ensemble models, with the possibility to include: 
    * Support Vector Machines
    * Decision Tree Classifiers
    * k-Nearest Neighbours Classifiers
* Cross-validation
* Confusion matrices

## Before starting
To use with Jupyter Notebooks be sure to install a Julia kernel. To access the functions `include("utils/ML1functions.jl")`.  
The "gene expression cancer RNA-Seq Data Set" is available at https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq. Save the `data.csv` and `labels.csv` under a folder named `dataset`. 