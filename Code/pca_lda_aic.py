from lda import LDA
from pca import PCA
from AIC import AIC
from dataset import train_features,train_labels,test_features, write_results
import numpy as np
import matplotlib.pyplot as plt


num_component = np.unique(train_labels).shape[0]
num_feature = 112

preprocesser_kwargs = {'num_feature': num_feature}
model_kwargs = {'num_component': num_component, 'num_feature': num_feature}
validator = AIC(PCA, LDA, preprocesser_kwargs, model_kwargs, noise=0.01)

score = validator.validate(train_features, train_labels)
print( 'AIC is: ', score)