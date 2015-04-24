import numpy as np
import matplotlib as mpl # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

# load data
np.random.seed(5)
d =  datasets.load_digits()
X, y = d.data, d.target
labels = set(y)

# run pca
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

# draw plot
plt.clf()
plt.cla()

for label in labels:
    plt.scatter(X[y==label,0], X[y==label,1], c=mpl.cm.jet(1.*label/len(labels)), label=label)

plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()

plt.savefig('mnist.png')
