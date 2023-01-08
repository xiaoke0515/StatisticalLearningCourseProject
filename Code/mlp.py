from dataset import train_features,train_labels,test_features, write_results
from pca import PCA
import numpy as np
from sklearn.neural_network import MLPClassifier

class MLP:
    def __init__(self, num_class = 0, num_feature = 0, num_hidden=100):
        self.num_feature = num_feature
        self.num_class = num_class
        self.num_hidden = num_hidden
        self.mlp = MLPClassifier(hidden_layer_sizes=num_hidden)

    def fit(self, features, labels):
        assert len(features.shape) == 2 and len(labels.shape) == 1
        assert features.shape[0] == labels.shape[0]

        self.mlp.fit(features, labels)

    def predict(self, features):
        return self.mlp.predict(features)

    def __len__(self):
        num = 0
        num += self.num_feature * self.num_hidden
        num += self.num_hidden * self.num_class
        return num


def train():
    #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_1
    num_component = np.unique(train_labels).shape[0]
    num_feature = 100#train_features.shape[1]

    pca = PCA(num_feature)
    pca.fit(train_features)
    train_features_reduced = pca.predict(train_features)

    #model = LDA_1()#(num_component, num_feature)
    model = MLP(num_component, num_feature, num_hidden=200)

    model.fit(train_features_reduced, train_labels)

    predicts = model.predict(pca.predict(test_features))
    print(np.unique(predicts))

    write_results(predicts, file_name='result_mlp_pca.csv')
    #write_results(predicts, file_name='result_LDA_sklearn.csv')

if __name__ == '__main__':
    train()