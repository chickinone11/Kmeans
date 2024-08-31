import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
import time
#Sử dụng thư viện sẽ tốn nhiều thời gian hơn code 
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

Z = linkage(data, method='ward') 

max_d = 3.0  

start_time = time.time()
clusters = fcluster(Z, max_d, criterion='distance') 

kmeans = KMeans(n_clusters=len(np.unique(clusters)), random_state=42, n_init=10)
kmeans.fit(data)
end_time = time.time()

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Time: " + str(end_time - start_time))
print("Labels:", labels)
print("Centroids:\n", centroids)