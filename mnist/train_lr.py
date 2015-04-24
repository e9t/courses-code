#! /usr/bin/python
# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
from sklearn import cross_validation, datasets
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


# load data
d = datasets.fetch_mldata('MNIST original')
X, y = d['data'], d['target']
print(X.shape, y.shape) # returns (70000, 784) (70000,)

# data exploration
X0 = X[0].reshape(28, 28)           # reshape 1*784 array to 28*28 array
plt.rc('image', cmap='binary')      # set runtime configurations (rc) for color maps
plt.matshow(X0)                     # plot matrix
plt.savefig('X0.png')               # save plot to image file

# data partitioning
X_train, X_test, y_train, y_test =\
        cross_validation.train_test_split(X, y, test_size=0.4, random_state=1234)

# training
lr = LogisticRegression(random_state=1234)      # LR instance 생성
lr.fit(X_train, y_train)                        # LR 학습 (takes almost an hour)
joblib.dump(lr, 'lr_randomstate_1234.pkl')
