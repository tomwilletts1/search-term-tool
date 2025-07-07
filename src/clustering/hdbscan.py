import hdbscan

def run_hdbscan(embeddings):
    cluster = hdbscan.HDBSCAN()
    labels = cluster.fit_predict(embeddings)
    return labels 