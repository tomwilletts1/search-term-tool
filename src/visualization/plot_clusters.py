import umap
import matplotlib.pyplot as plt

def plot_clusters(embeddings, labels):
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(embedding_2d[:,0], embedding_2d[:,1], c=labels, cmap='tab10')
    ax.set_title('Cluster Visualization')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    if hasattr(scatter, 'get_array'):
        fig.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    return fig