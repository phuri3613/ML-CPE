from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbors(X, n_neighbors=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices