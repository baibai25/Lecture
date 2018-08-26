import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def find_k(X):
    # clustering with k=1, 2,..., 10
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='random', max_iter=300, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # plot elbow graph
    plt.figure()
    plt.plot(range(1, 11), wcss, 'bx-')
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    plt.xticks(np.arange(0, 11, 1.0))
    plt.grid()
    plt.savefig("elbow.png",format = 'png', dpi=300)

def clustering(X, y):
    # clustering
    kmeans = KMeans(n_clusters=3, init='random', max_iter=300, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    # calculate adjusted rand score
    ars = round(adjusted_rand_score(y, y_kmeans), 3)
    print('ars={}'.format(ars))
    
    # plot clusters
    plt.figure()
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='green', label='Iris-versicolour')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='red', label='Iris-setosa')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='blue', label='Iris-virginica')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=100, c='yellow', label = 'Centroids')
    plt.annotate('ARS={}'.format(ars), xy=(7.5, 2.0))
    plt.legend()
    plt.savefig("cluster.png",format = 'png', dpi=300)
    
if __name__ == '__main__':
    
    # load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    find_k(X)
    clustering(X, y)
