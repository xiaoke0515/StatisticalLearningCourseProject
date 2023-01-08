from dataset import train_features,train_labels,test_features, write_results
from pca import PCA
import numpy as np

class LDA:
    def __init__(self, num_component = 0, num_feature = 0):
        self.pi = np.zeros([num_component])
        self.mu = np.zeros([num_component, num_feature])
        self.sigma = 0
        self.sigma_inv = 0
        self.label_unique = np.zeros([num_component])
        self.num_component = num_component

    def fit(self, features, labels):
        assert len(features.shape) == 2 and len(labels.shape) == 1
        assert features.shape[0] == labels.shape[0]

        self.label_unique = np.unique(labels)
        assert self.label_unique.shape[0] == self.num_component
        sample_num = labels.shape[0]
        for i, label in enumerate(self.label_unique):
            sample = features[labels==label, :]
            self.pi[i] = sample.shape[0] / sample_num
            self.mu[i] = np.sum(sample, axis=0) / sample.shape[0]
            self.sigma += (sample - self.mu[i]).transpose() @ (sample - self.mu[i])
        self.sigma = self.sigma / (sample_num - self.num_component)
        self.sigma_inv = np.linalg.inv(self.sigma)

    def predict(self, features):
        delta = features @ self.sigma_inv @ self.mu.transpose() - 0.5 * np.diag(self.mu @ self.sigma_inv @ self.mu.transpose()) + np.log(self.pi)
        predicts = np.argmax(delta, axis=1)
        return predicts

    def __len__(self):
        num = 0
        num += np.prod(self.pi.shape)
        num += np.prod(self.mu.shape)
        num += np.prod(self.sigma.shape)
        return num


def train():
    #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_1
    num_component = np.unique(train_labels).shape[0]
    num_feature = 100#train_features.shape[1]

    pca = PCA(num_feature)
    pca.fit(train_features)
    train_features_reduced = pca.predict(train_features)

    #model = LDA_1()#(num_component, num_feature)
    model = LDA(num_component, num_feature)

    model.fit(train_features_reduced, train_labels)

    predicts = model.predict(pca.predict(test_features))
    print(np.unique(predicts))

    write_results(predicts, file_name='result_LDA_pca.csv')
    #write_results(predicts, file_name='result_LDA_sklearn.csv')

if __name__ == '__main__':
    train()