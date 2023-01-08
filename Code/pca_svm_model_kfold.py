from svm import SVM
from pca import PCA
from kfold_validation import KFoldCrossValidation
from dataset import train_features,train_labels,test_features, write_results
import numpy as np

for order in [1,2,3]:
    num_class = np.unique(train_labels).shape[0]
    num_feature = 112

    preprocesser_kwargs = {'num_feature': num_feature}
    model_kwargs = {'num_class': num_class, 'num_feature': num_feature, 'dim': order}
    validator = KFoldCrossValidation(PCA, SVM, 10, preprocesser_kwargs, model_kwargs)

    score = validator.validate(train_features, train_labels)
    print('K-Fold Cross Validation Score is: ', score)

    preprocesser = PCA(**preprocesser_kwargs)
    preprocesser.fit(train_features)
    train_features_reduced = preprocesser.predict(train_features)
    model = SVM(**model_kwargs)
    model.fit(train_features_reduced, train_labels)

    test_features_reduced = preprocesser.predict(test_features)
    predicts = model.predict(test_features_reduced)
    print(np.unique(predicts))

    write_results(predicts, file_name='result_pca_112_SVM_%d.csv' % order)
    #write_results(predicts, file_name='result_LDA_sklearn.csv')