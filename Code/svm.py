from dataset import train_features,train_labels,test_features, write_results
import numpy as np
import cvxopt as cvx
from pca import PCA
cvx.solvers.options['show_progress'] = False


class SVM:

    def __init__(self, num_class=0, num_feature=0, kernel='poly', dim=3):
        assert kernel in ['poly']
        self.kernel_dim = dim
        self.kernel = kernel
        self.num_feature = num_feature
        self.num_class = num_class

    def fit(self, features, labels):
        assert len(features.shape) == 2 and len(labels.shape) == 1
        assert features.shape[0] == labels.shape[0]

        self.label_unique = np.unique(labels)
        assert self.label_unique.shape[0] == self.num_class
        sample_num = labels.shape[0]

        # train classifiers
        self.classifiers = []
        for i in range(self.num_class):
            self.classifiers.append([])
            samples_class_i = features[labels == i, :]
            num_class_i = samples_class_i.shape[0]
            for j in range(self.num_class):
                if j <= i:
                    continue
                else:
                    print('\rtraining the %dth/%d svm model' % (i * (i + self.num_class) / 2 + j-i, self.num_class * (self.num_class - 1) / 2), end='')
                    #print('\n')
                    samples_class_j = features[labels == j, :]
                    num_class_j = samples_class_j.shape[0]
                    samples_train = np.concatenate([samples_class_i, samples_class_j], axis=0)
                    labels_bin = np.concatenate([np.ones([num_class_i]), - np.ones([num_class_j])])
                    alpha_i_j = self._solve_lagrange_multiplier(samples_train, labels_bin)
                    support_alpha = alpha_i_j[alpha_i_j > 1e-6]
                    support_vector = samples_train[alpha_i_j > 1e-6]
                    support_label = labels_bin[alpha_i_j > 1e-6]
                    #print(support_alpha.shape, support_vector.shape, support_label.shape )
                    self.classifiers[i].append([support_alpha, support_vector, support_label])
        print('\n')


    def predict(self, features):
        num_sample = features.shape[0]
        voter = np.zeros([num_sample, self.num_class])
        for i in range(self.num_class):
            for j in range(self.num_class):
                if j <= i:
                    continue
                alpha, vector, label = self.classifiers[i][j - i - 1]
                K = self._construct_kernel_mat(features, vector)
                y_pred = np.sign((alpha * label) @ K.transpose())
                #vote
                y_pred[y_pred < 0] = 0
                voter[:, i] += y_pred
                voter[:, j] += (1 - y_pred)
        label = np.argmax(voter, axis=1)
        return label

    def _kernel(self, a, b):
        if self.kernel == 'poly':
            return (1 + a @ b.transpose()) ** self.kernel_dim

    def _construct_kernel_mat(self, features, support_features):
        num_sample = features.shape[0]
        num_kernel = support_features.shape[0]
        K = np.zeros([num_sample, num_kernel])
        for i, x1 in enumerate(features):
            for j, x2 in enumerate(support_features):
                K[i, j] = self._kernel(x1, x2)
        return K

    def _solve_lagrange_multiplier(self, features, labels_bin):
        sample_num = labels_bin.shape[0]
        K = self._construct_kernel_mat(features, features)

        #compute parameters
        # cvx's qp solver:
        # min 1/2*xT@P@X + qT@X
        # s.t.
        # G@X <= h
        # A@X = b

        # SVM's qp problem:
        # max -1/2*alphaT@(y@yT * K(x,x))@alpha + alpha
        # s.t.
        # alpha >= 0
        # alphaTy = 0

        P = labels_bin.transpose() @ labels_bin * K
        q = -1 * np.ones_like(labels_bin)
        G =  -np.eye(sample_num)
        h = np.zeros([sample_num, 1])
        A = labels_bin.reshape([1, -1])
        b = np.zeros([1])

        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(b)

        solver = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = solver['x']
        return np.array(alpha).reshape([-1])


    def __len__(self):
        num = 0
        for i in range(self.num_class):
            for j in range(self.num_class):
                if j <= i:
                    continue
                alpha, vector, label = self.classifiers[i][j - i - 1]
                num += np.prod(alpha.shape)
                num += np.prod(vector.shape)
                num += np.prod(label.shape)
        return num

def train():

    num_class = np.unique(train_labels).shape[0]
    num_feature =112 
    pca = PCA(num_feature)
    pca.fit(train_features)
    train_features_reduced = pca.predict(train_features)

    model = SVM(num_class, num_feature, kernel='poly', dim=3)

    model.fit(train_features_reduced, train_labels)

    predicts = model.predict(pca.predict(test_features))
    print(np.unique(predicts))

    write_results(predicts, file_name='result_SVM_poly_dim3.csv')

if __name__ == '__main__':
    train()