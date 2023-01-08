import numpy as np
from dataset import train_features, train_labels, test_features
from pca import PCA

class AIC:
    def __init__(self, preprocesser, model, preprocesser_kwargs, model_kwargs, noise = 0.01):
        self.preprocesser = preprocesser
        self.model = model
        self.preprocesser_kwargs = preprocesser_kwargs
        self.model_kwargs = model_kwargs
        self.noise = noise

    def validate(self, train_features, train_labels):
        num_sample = train_features.shape[0]
        rand_sort = np.argsort(np.random.random([num_sample]))
        #rand_sort = np.arange(num_sample)
        scores = []
        if not self.preprocesser == None:
            preprocesser = self.preprocesser(**(self.preprocesser_kwargs))
        model = self.model(**(self.model_kwargs))
        if not self.preprocesser == None:
            if self.preprocesser == PCA:
                preprocesser.fit(train_features)
            else:
                preprocesser.fit(train_features * (1 + np.random.normal(loc=0.0, scale=self.noise, size=train_features.shape)), train_labels)
            train_set_feat = preprocesser.predict(train_features)

        model.fit(train_set_feat, train_labels)
        train_res = model.predict(train_set_feat)
        AIC = np.sum(train_res == train_labels) / num_sample + 2 * len(model) / num_sample * (self.noise ** 2)
        return AIC
