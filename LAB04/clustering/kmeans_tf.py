from sklearn.cluster import KMeans

def run_kmeans(X, max_k=10):
    inertias = []
    models = {}
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        models[k] = kmeans
        
    return inertias, models