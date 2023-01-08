from mlp import MLP
from pca import PCA
from AIC import AIC
from dataset import train_features,train_labels,test_features, write_results
import numpy as np

for num_hidden in [100,200,300, 400, 500, 600]:
    num_class = np.unique(train_labels).shape[0]
    num_feature = 112

    preprocesser_kwargs = {'num_feature': num_feature}
    model_kwargs = {'num_class': num_class, 'num_feature': num_feature, 'num_hidden': num_hidden}
    validator = AIC(PCA, MLP, preprocesser_kwargs, model_kwargs, noise=0.001)

    score = validator.validate(train_features, train_labels)
    print('AIC is: ', score)
