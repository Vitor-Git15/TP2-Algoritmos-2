from KCenters import *
from DatasetReader import *

import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

class DataAnalyzer:
    def __init__(self):
        self.data_reader = DatasetReader()
        self.kcenters = KCenters()
        self.options = {
            0 :   "datasets/breast_cancer/",
            1 :   "datasets/blood+transfusion+service+center/",
            2 :   "datasets/wine+quality/",
            3 :   "datasets/banknote+authentication/",
            4 :   "datasets/shill+bidding+dataset/",
            5 :   "datasets/spambase/",
            6 :   "datasets/balance+scale/",
            7 :   "datasets/seoul+bike+sharing+demand/",
            8 :   "datasets/iranian+churn+dataset/",
            9 :   "datasets/facebook+live+sellers+in+thailand/",
        }

    def run_data_analyzer(self, opt):
        data, labels = self.data_reader.read_data(opt)
        k = np.unique(labels).size
        p = 2
        out_file = open(self.options.get(opt, '') + 'analyses_result.txt', 'w')

        self.print_results_header(out_file)
        
        radius_manhattan, radius_euclidean, silhouettes, rands, execution_time = self.calculate_metrics(data, labels, k, p, out_file)
        self.print_metrics(radius_manhattan, radius_euclidean, silhouettes, rands, execution_time, out_file)

        self.print_kmeans_results(data, labels, out_file)

        out_file.close()

    def print_results_header(self, out_file):
        print("Silhouette \t Rand \t Radius (Manhattan) \t Radius (Euclides) \t Execution Time", file=out_file)

    def calculate_metrics(self, data, labels, k, p, out_file):
        manhattan_metric = 1
        euclidean_metric = 2

        radius_manhattan = np.empty(30)
        radius_euclidean = np.empty(30)
        silhouettes = np.empty(30)
        rands = np.empty(30)
        execution_time = np.empty(30)
        
        self.kcenters.points = data
        weights = self.kcenters.distance_matrix(p)
        self.kcenters.weights = weights
        
        for i in range(30):
            start_time = time.time()
            centroids = self.kcenters.fit(k, p)
            predictions = self.kcenters.predict(centroids, p)
            approx_silhouette = silhouette_score(data, predictions)
            approx_rand = rand_score(labels, predictions)
            approx_radius_manhattan = self.kcenters.radius(centroids, manhattan_metric)
            approx_radius_euclidean = self.kcenters.radius(centroids, euclidean_metric)
            end_time = time.time()

            radius_manhattan[i] = approx_radius_manhattan
            radius_euclidean[i] = approx_radius_euclidean
            silhouettes[i] = approx_silhouette
            rands[i] = approx_rand
            execution_time[i] = end_time - start_time

            print(f"{approx_silhouette}\t{approx_rand}\t{approx_radius_manhattan}\t{approx_radius_euclidean}\t{execution_time[i]}", file=out_file)
        print("\n")
        return radius_manhattan, approx_radius_euclidean, silhouettes, rands, execution_time

    def print_metrics(self, radius_manhattan, radius_euclidean, silhouettes, rands, execution_time, out_file):
        print("\n", file=out_file)
        print((100 * '-'), file=out_file)
        print("\nK-Centers Results", file=out_file)
        print((100 * '-'), file=out_file)
        print(f"Max Radius (Manhattan) [Mean] = {radius_manhattan.mean()}\t [Standard Deviation] = {radius_manhattan.std()}", file=out_file)
        print(f"Max Radius (Euclidean) [Mean] = {radius_euclidean.mean()}\t [Standard Deviation] = {radius_euclidean.std()}", file=out_file)
        print(f"Silhouette [Mean] = {silhouettes.mean()}\t [Standard Deviation] = {silhouettes.std()}", file=out_file)
        print(f"Rand [Mean] = {rands.mean()}\t [Standard Deviation] = {rands.std()}", file=out_file)
        print(f"Execution Time [Mean] = {execution_time.mean()}\t [Standard Deviation] = {execution_time.std()}", file=out_file)

        print((100 * '-'), file=out_file)

    def print_kmeans_results(self, data, labels, out_file):
        print("\nK-Means Results", file=out_file)
        print((100 * '-'), file=out_file)
        k = np.unique(labels).size
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(data)
        kmeans_centers = kmeans.cluster_centers_
        kmeans_predictions = kmeans.predict(data)
        start_time = time.time()
        manhattan_dists = manhattan_distances(data, kmeans_centers)
        euclidean_dists = euclidean_distances(data, kmeans_centers)
        max_radius_manhattan = np.max(manhattan_dists)
        max_radius_euclidean = np.max(euclidean_dists)

        kmeans_silhouette = silhouette_score(data, kmeans_predictions)
        kmeans_rand = rand_score(labels, kmeans_predictions)
        end_time = time.time()

        print("Max Radius (Manhattan) = ", max_radius_manhattan, file=out_file)
        print("Max Radius (Euclidean) = ", max_radius_euclidean, file=out_file)
        print("Silhouette = ", kmeans_silhouette, file=out_file)
        print("Rand = ", kmeans_rand, file=out_file)
        print("Execution Time = ", (end_time - start_time), file=out_file)

data_analyzer = DataAnalyzer()
[data_analyzer.run_data_analyzer(i) for i in range(10)]

