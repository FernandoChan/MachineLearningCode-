import scipy
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import datasets
digits= datasets.load_digits()

plt.figure(1)
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation = 'nearest')
plt.show()

iris = datasets.load_iris()
from sklearn.decomposition import PCA

Xiris = iris.data[:, :3]
yiris = iris.target


x_min , x_max = Xiris[:, 0].min() - .5,Xiris[:, 0].max() + .5
y_min , y_max = Xiris[:, 1].min() - .5 , yiris[:,1].max() +.5
plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(Xiris[:, 0], Xiris[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim (x_min, x_max)
plt.ylim (y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
