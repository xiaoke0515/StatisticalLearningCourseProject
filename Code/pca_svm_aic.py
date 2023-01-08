from svm import SVM
from pca import PCA
from AIC import AIC
from dataset import train_features,train_labels,test_features, write_results
import numpy as np

for order in [1,2,3]:
    num_class = np.unique(train_labels).shape[0]
    num_feature = 112

    preprocesser_kwargs = {'num_feature': num_feature}
    model_kwargs = {'num_class': num_class, 'num_feature': num_feature, 'dim': order}
    validator = AIC(PCA, SVM, preprocesser_kwargs, model_kwargs, noise=0.01)

    score = validator.validate(train_features, train_labels)
    print('AIC Score is: ', score)
