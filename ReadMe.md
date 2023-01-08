# Introduction

This is code for the CS7304H (2022 Fall) course project.
The task is to fit a model on the training dataset and predict on a noised test dataset.
The code try two methods to reduce the data dimension, PCA and Reduce Rank LDA, and three models, LDA, SVM and MLP.
K-Fold cross validation and AIC are used to evaluate the models.

# Dependency

```
python=3.9.15
matplotlib
sklearn
numpy
```

# Usage
Put dataset in the *dataset* folder.

* Figure 1: File *pca_components.pdf*
```
python3 pca_research.py
```

* Figure 2: File *rrlda_lda_cv_noise.pdf*
```
python3 rrlda_lda_model_kfold.py
```

* Figure 3: File *pca_lda_cv_noise.pdf*
```
python3 pca_lda_model_kfold.py
```

* Table 2 - CV & Figure 4 & Figure 5: 

File *result_pca_112_LDA.csv*
```
python3 pca_lda_model_kfold.py
```
File *result_pca_112_SVM_1.csv*\
File *result_pca_112_SVM_2.csv*\
File *result_pca_112_SVM_3.csv*
```
python3 pca_svm_model_kfold.py
```
File *result_pca_112_MLP_100.csv*\
File *result_pca_112_MLP_200.csv*\
File *result_pca_112_MLP_300.csv*\
File *result_pca_112_MLP_400.csv*
```
python3 pca_mlp_model_kfold.py
```
File *result_rrlda_19_LDA.csv*
```
python3 rrlda_lda_cv_model_kfold.py
```
* Table 2 - AIC:
```
python3 pca_lda_aic.py
python3 pca_svm_aic.py
python3 pca_mlp_aic.py
```