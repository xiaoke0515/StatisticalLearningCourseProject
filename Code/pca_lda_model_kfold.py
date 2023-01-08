from lda import LDA
from pca import PCA
from kfold_validation import KFoldCrossValidation
from dataset import train_features,train_labels,test_features, write_results
import numpy as np
import matplotlib.pyplot as plt


noise_range = np.arange(0, 0.001, 0.0001)
cv_res = []
for noise in noise_range:
    num_component = np.unique(train_labels).shape[0]
    num_feature = 112

    preprocesser_kwargs = {'num_feature': num_feature}
    model_kwargs = {'num_component': num_component, 'num_feature': num_feature}
    validator = KFoldCrossValidation(PCA, LDA, 10, preprocesser_kwargs, model_kwargs, val_noise=noise)

    score = validator.validate(train_features, train_labels)
    print('Noise variance: ', noise, 'K-Fold Cross Validation Score is: ', score)
    cv_res.append(score)


plt.plot(noise_range, cv_res, 'o')
plt.ylabel('Cross Validation')
plt.xlabel('noise')
plt.savefig('pca_lda_cv_noise.pdf')

preprocesser = PCA(**preprocesser_kwargs)
preprocesser.fit(train_features)
train_features_reduced = preprocesser.predict(train_features)
model = LDA(**model_kwargs)
model.fit(train_features_reduced, train_labels)

test_features_reduced = preprocesser.predict(test_features)
predicts = model.predict(test_features_reduced)
print(np.unique(predicts))

write_results(predicts, file_name='result_pca_112_LDA.csv')
#write_results(predicts, file_name='result_LDA_sklearn.csv')