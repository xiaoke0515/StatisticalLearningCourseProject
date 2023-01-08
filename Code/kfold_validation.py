import numpy as np
from dataset import train_features, train_labels, test_features
from pca import PCA

class KFoldCrossValidation:
    def __init__(self, preprocesser, model, k, preprocesser_kwargs, model_kwargs, val_noise = 0):
        self.preprocesser = preprocesser
        self.model = model
        assert isinstance(k, int)
        assert k > 2
        self.k = k
        self.preprocesser_kwargs = preprocesser_kwargs
        self.model_kwargs = model_kwargs
        self.val_noise = val_noise

    def validate(self, train_features, train_labels):
        num_sample = train_features.shape[0]
        rand_sort = np.argsort(np.random.random([num_sample]))
        #rand_sort = np.arange(num_sample)
        scores = []
        for i in range(self.k):
            if not self.preprocesser == None:
                preprocesser = self.preprocesser(**(self.preprocesser_kwargs))
            model = self.model(**(self.model_kwargs))
            train_set_feat = np.concatenate([train_features[rand_sort[:int(i / self.k * num_sample)], :], train_features[rand_sort[int((i + 1) / self.k * num_sample):], :]], axis=0)
            train_set_lab = np.concatenate([train_labels[rand_sort[:int(i / self.k * num_sample)]], train_labels[rand_sort[int((i + 1) / self.k * num_sample):]]], axis=0)
            if not self.preprocesser == None:
                if self.preprocesser == PCA:
                    preprocesser.fit(train_set_feat)
                else:
                    preprocesser.fit(train_set_feat, train_set_lab)
                train_set_feat = preprocesser.predict(train_set_feat)
            val_set_feat = train_features[rand_sort[int(i / self.k * num_sample): int((i + 1) / self.k * num_sample)]]
            val_set_feat *= 1 + np.random.normal(loc=0, scale=self.val_noise, size=val_set_feat.shape)
            if not self.preprocesser == None:
                val_set_feat = preprocesser.predict(val_set_feat)
            val_set_lab = train_labels[rand_sort[int(i / self.k * num_sample): int((i + 1) / self.k * num_sample)]]

            model.fit(train_set_feat, train_set_lab)
            val_result = model.predict(val_set_feat)
            #print(np.unique(val_result))
            score = np.sum(val_result == val_set_lab) / val_set_lab.shape[0]
            scores.append(score)
        return np.mean(scores)
