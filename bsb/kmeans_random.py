import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

class KMeans(object):

    def __init__(self, n_clusters=2, max_iter=300):
        
        self.n_clusters = n_clusters    # number of cluster
        self.max_iter = max_iter    # max iterration
        self.cluster_centers = None

    def fit_predict(self, features):
        # select initial value of centroid
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers = features[initial_centroid_indexes]
        pred = np.zeros(features.shape)

        # update cluster
        for _ in range(self.max_iter):
            # update labels (minimum distance)
            new_pred = np.array([
                np.array([
                    self.euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers
                ]).argmin()
                for p in features
            ])
            
            # if new pred == previous pred 
            if np.all(new_pred == pred):
                break
            pred = new_pred

            # recalculate centroid
            self.cluster_centers = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])
        return pred

    def euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)


def main():
    N_CLUSTERS = 5

    # generate isotropic gaussian blobs
    dataset = datasets.make_blobs(centers=N_CLUSTERS)
    features = dataset[0]

    # clustering
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    # plot cluster
    for i in range(N_CLUSTERS):
        labels = features[pred == i]
        plt.scatter(labels[:, 0], labels[:, 1])
    centers = cls.cluster_centers
    plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')
    plt.savefig("kmeans.png",format = 'png', dpi=300)

if __name__ == '__main__':
    main()
