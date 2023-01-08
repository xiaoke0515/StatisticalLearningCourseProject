from dataset import train_features,train_labels,test_features, write_results
import numpy as np
import scipy

class ReducedRankLDA:

    def __init__(self, num_component = 0, num_feature = 0):
        self.pi = np.zeros([num_component])
        self.mu = np.zeros([num_component, num_feature])
        self.sigma = 0
        self.sigma_inv = 0
        self.V = 0
        self.label_unique = np.zeros([num_component])
        self.num_component = num_component
        self.num_feature = num_feature

    def fit(self, features, labels):
        assert len(features.shape) == 2 and len(labels.shape) == 1
        assert features.shape[0] == labels.shape[0]
        total_feature_num = features.shape[1]
        self.mu = np.zeros([self.num_component, total_feature_num])

        self.label_unique = np.unique(labels)
        assert self.label_unique.shape[0] == self.num_component
        sample_num = labels.shape[0]
        for i, label in enumerate(self.label_unique):
            sample = features[labels==label, :]
            self.pi[i] = sample.shape[0] / sample_num
            self.mu[i] = np.sum(sample, axis=0) / sample.shape[0]
            self.sigma += (sample - self.mu[i]).transpose() @ (sample - self.mu[i])
        self.sigma = self.sigma / (sample_num - self.num_component)
        #self.sigma_inv = np.linalg.inv(self.sigma)

        #reduce rank
        W_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(self.sigma))
        M_star = self.mu @ W_inv_sqrt
        B_star = np.cov(M_star.transpose())
        evals, V_star = np.linalg.eig(B_star)
        self.V = W_inv_sqrt @ V_star[:, :self.num_feature]


    def predict(self, features):
        #delta = features @ self.sigma_inv @ self.mu.transpose() - 0.5 * np.diag(self.mu @ self.sigma_inv @ self.mu.transpose()) + np.log(self.pi)
        reduced = features @ self.V
        return np.real(reduced)


def train():
    num_component = np.unique(train_labels).shape[0]
    num_feature = train_features.shape[1]
    model = ReducedRankLDA(num_component, num_feature)

    model.fit(train_features, train_labels)

    predicts = model.predict(test_features)
    print(np.unique(predicts))

    write_results(predicts, file_name='result_ReducedRankLDA.csv')

if __name__ == '__main__':
    train()