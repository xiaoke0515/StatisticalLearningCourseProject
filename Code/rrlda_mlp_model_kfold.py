from mlp import MLP
from rr_lda import ReducedRankLDA
from kfold_validation import KFoldCrossValidation
from dataset import train_features,train_labels,test_features, write_results
import numpy as np

for num_hidden in [20,40,60, 80]:
    num_class = np.unique(train_labels).shape[0]
    num_feature = 19

    preprocesser_kwargs = {'num_component': num_class, 'num_feature': num_feature}
    model_kwargs = {'num_class': num_class, 'num_feature': num_feature, 'num_hidden': num_hidden}
    validator = KFoldCrossValidation(ReducedRankLDA, MLP, 10, preprocesser_kwargs, model_kwargs)

    score = validator.validate(train_features, train_labels)
    print('K-Fold Cross Validation Score is: ', score)

    preprocesser = ReducedRankLDA(**preprocesser_kwargs)
    preprocesser.fit(train_features, train_labels)
    train_features_reduced = preprocesser.predict(train_features)
    model = MLP(**model_kwargs)
    model.fit(train_features_reduced, train_labels)

    test_features_reduced = preprocesser.predict(test_features)
    predicts = model.predict(test_features_reduced)
    print(np.unique(predicts))

    write_results(predicts, file_name='result_rrlda_19_MLP_%d.csv' % num_hidden)
    #write_results(predicts, file_name='result_LDA_sklearn.csv')