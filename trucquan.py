import numpy as np
import matplotlib.pyplot as plt
import time

class KMeans:
    def __init__(self, epsilon: float = 1e-5, maxiter: int = 10000):
        self._epsilon = epsilon
        self._maxiter = maxiter

    def update_labels(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dis = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dis, axis=1)  

    def kmeans(self, data: np.ndarray, C: int, seed: int = 0) -> tuple:
        if seed > 0:
            np.random.seed(seed=seed)
        centroids = data[np.random.choice(len(data), C, replace=False)] 
        for step in range(self._maxiter):
            labels = self.update_labels(data, centroids)
            new_c = np.array([data[labels == k].mean(axis=0) for k in range(C)])
            if np.sum((new_c - centroids) ** 2) < self._epsilon:
                break
            centroids = new_c
        return labels, centroids, step + 1


data = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0],
])

start_time = time.time()
kmeans = KMeans(epsilon=1e-5, maxiter=10000)
labels, centroids, n_iter = kmeans.kmeans(data, C=3, seed=42)
end_time = time.time()

print("Time: " + str(end_time - start_time))
print("Labels:", labels)
print("Centroids:\n", centroids)
print("Number iterations:", n_iter)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for k in range(3):
    plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], 
                color=colors[k], label=f'Cluster {k + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
