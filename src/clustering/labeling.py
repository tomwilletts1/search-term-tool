from collections import Counter

def label_clusters(terms, labels):
    cluster_labels = {}
    for cluster_id in set(labels):
        cluster_terms = [terms[i] for i in range(len(terms)) if labels[i] == cluster_id]
        words = ' '.join(cluster_terms).split()
        most_common = Counter(words).most_common(1)
        label = most_common[0][0] if most_common else f"Cluster {cluster_id}"
        cluster_labels[cluster_id] = label
    return cluster_labels 