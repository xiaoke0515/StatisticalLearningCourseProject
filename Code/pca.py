from dataset import train_features,train_labels,test_features
import numpy as np

class PCA:

    def __init__(self, num_feature = 0):
        self.num_feature = num_feature
        self.P = 0

    def fit(self, features):
        features_centered = features - np.mean(features, axis=0)
        cov = np.cov(features_centered.transpose())
        _, evec = np.linalg.eig(cov)
        self.P = evec[:, :self.num_feature]

    def predict(self, features):
        return features @ self.P
    
    def plot_error_feature(self, features, thresholds):
        features_centered = features - np.mean(features, axis=0)
        cov = np.cov(features_centered.transpose())
        #_, S, _ = np.linalg.svd(cov)
        S, _ = np.linalg.eig(cov)
        accumulated_error = 1 - np.cumsum(S) / np.sum(S)

        #thresholds = [0.9, 0.95, 0.99]
        num_components = []
        for th in thresholds:
            num_components.append(np.min(np.where (accumulated_error < 1 - th)))
        print('thresholds:', thresholds)
        print('the num components are: ', num_components)
        import matplotlib.pyplot as plt 
        plt.plot(np.arange(accumulated_error.shape[0]), accumulated_error)
        for th, num in zip(thresholds, num_components):
            plt.plot([0, num], [1-th, 1-th], '--b')
            plt.plot([num, num], [0, 1-th], '--b')
            plt.text(0, 1 - th, 'Error=%.02f' % (1 - th))
            plt.text(num, 0, 'Num. of Components=%d' % num)

        plt.xticks([])
        plt.xlabel('Number of components')
        plt.ylabel('Error')
        plt.savefig('pca_components.pdf')