import numpy as np

class KCenters:
    def __init__(self):
        self.points = None
        self.weights = None

    def _minkowski_distance(self, p1, p2, p):
        
        if len(p1) != len(p2):
          raise ValueError("The points must have the same dimension.")
        
        diff = np.abs(np.array(p1) - np.array(p2))
        dist = np.sum(np.power(diff, p))**(1/p)
        return dist

    def _clustering(self, k):
      n = len(self.points)

      if k >= n:
          return self.points

      C = np.empty((k,) + self.points.shape[1:])
      minDist = np.full(n, np.inf)
      currCenter = np.random.randint(n)

      for i in range(k):
          C[i] = self.points[currCenter]
          minDist = np.minimum(minDist, self.weights[currCenter])
          currCenter = np.argmax(minDist)

      return C

    def _predict(self, centroids, p):
        minDist = np.array([[self._minkowski_distance(point, centroid, p) for centroid in centroids] for point in self.points])
        predictions = np.argmin(minDist, axis=1)

        return predictions

    def _radius(self, centroids, p):
        radius = np.max([np.max([self._minkowski_distance(point, centroid, p) for centroid in centroids]) for point in self.points])
        return radius
    
    def distance_matrix(self, p):
        n = len(self.points)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            matrix[i] = [self._minkowski_distance(self.points[i], point, p) for point in self.points]

        return matrix

    def fit(self, k, p):
        return self._clustering(k)

    def predict(self, centroids, p):
        return self._predict(centroids, p)

    def radius(self, centroids, p):
        return self._radius(centroids, p)
