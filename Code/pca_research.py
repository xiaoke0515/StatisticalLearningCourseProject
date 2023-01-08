from dataset import train_features,train_labels,test_features
import numpy as np
from pca import PCA


num_feature = 100
pca = PCA(num_feature)
pca.plot_error_feature(train_features, thresholds=[0.7])