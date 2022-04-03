import numpy as np
import pandas as pd
import random
import warnings
from numpy.linalg import norm


class KMeans:
    def __init__(self, k_count=10, similarity='eucledean', max_iterations=500):
        self.k_count = k_count
        self.max_iterations = max_iterations
        # Initializes the SSE to infinity for initial loop comparison.
        self.sse = np.inf
        self.iterations = 0
        if similarity == 'cosine':
            self.similarity = self._cosine_distance
        elif similarity == 'jaccard':
            self.similarity = self._jaccard_distance
        else:
            self.similarity = self._euclidean_distance

    def _k_means(self):
        # Initializes the SSE delta as negative to start loop.
        sse_delta = -1
        self.iterations = 0
        old_centroids = []

        #  Kmeans algorithm loop, continues until sse doesnt improve or reaches max iterations.
        while sse_delta <= 0 and not np.array_equal(old_centroids, self.centroids) and self.iterations < self.max_iterations:
            old_centroids = self.centroids.copy()
            # Assign closest cluster to sample based on closest cluster centroids.
            self.labels = self._assign_clusters(
                self.samples[:, None, :], self.centroids[None, :, :])

            # Find all points belonging to a cluster
            for i in range(self.k_count):
                # There will be warnings of division by
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cluster_members = self.samples[self.labels == i].mean(
                        axis=0)
                self.centroids[i] = cluster_members

            # Finds sse for all centroids
            current_sse = self._sum_square_error(
                self.samples.squeeze(), self.centroids.squeeze(), self.labels)
            # Checks if sse improved
            sse_delta = current_sse - self.sse
            self.sse = current_sse
            self.iterations += 1

    def _euclidean_distance(self, points1, points2):
        # Last dimention of array is always features, so checks the distance along them.
        distance = np.sqrt(
            ((points1 - points2) ** 2).sum(axis=points1.ndim - 1))
        return distance

    def _cosine_distance(self, points1, points2):
        # Takes dot product along last dimention of array then finds cosine distance.
        dot_product = np.tensordot(points1, points2, axes=(
            [points1.ndim - 1], [points2.ndim - 1]))
        norm1 = norm(points1, axis=points1.ndim - 1)
        norm2 = norm(points2, axis=points2.ndim - 1)
        distance = 1 - (dot_product.squeeze() / (norm1 * norm2))

        return distance

    def _jaccard_distance(self, points1, points2):
        numerator = (np.minimum(points1, points2)).sum(axis=points1.ndim - 1)
        denominator = (np.maximum(points1, points2)).sum(axis=points1.ndim - 1)
        distances = 1 - (numerator / denominator)
        return distances

    def _assign_clusters(self, points, centroids):
        # Finds distance from each point to all centroids.
        centroid_distances = self.similarity(points, centroids)
        # Calculates the closest centroid to each point.
        nearest_centroid = np.argmin(centroid_distances, axis=1)
        return nearest_centroid

    def _sum_square_error(self, samples, centroids, label_array):
        # For each sample, find distance to its assigned centroid divided by number of data points
        # We use euclidean distance for every SSE calculation
        sum_square_error = self._euclidean_distance(
            samples, centroids[label_array]).sum() / len(samples)
        return sum_square_error

    def fit(self, dataset):
        self.samples = dataset.to_numpy()
        self.labels = None
        # Set centroids, this round they are all random
        self.centroids = self.samples[random.sample(
            range(self.samples.shape[0]), self.k_count)]
        # Run kmeans algorithm
        self._k_means()
        self.dataset = dataset
        self.dataset['clusters'] = self.labels


if __name__ == "__main__":
    training_data = pd.read_csv('data.csv')
    nunique = training_data.nunique()
    cols_to_drop = nunique[nunique == 1].index
    test = training_data.drop(cols_to_drop, axis=1)
    km = KMeans(similarity='jaccard', k_count=10, max_iterations=50)
    km.fit(test)
    print(km.dataset['clusters'].value_counts())
    print(km.iterations)
    # test['clusters'] = km.labels
    # with open('test.txt', 'w') as filez:
    #     np.set_printoptions(threshold=np.inf)
    #     filez.write(str(km.dataset))
