from sklearn.cluster import KMeans

def run_kmeans(embeddings, n_clusters):
    cluster = KMeans(n_clusters=n_clusters, random_state=42)
    labels = cluster.fit_predict(embeddings)
    return labels 