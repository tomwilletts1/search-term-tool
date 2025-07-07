import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from src.embeddings import embed_texts
from src.clustering.kmeans import run_kmeans
from src.clustering.hdbscan import run_hdbscan
from src.services.openai_services import get_ai_cluster_analysis
from src.visualization.plot_clusters import plot_clusters

def main(input_csv, output_csv, term_column='search_terms', cluster_method='kmeans', n_clusters=3):
    """
    Orchestrates the workflow: reads CSV, generates hybrid embeddings, clusters, generates AI labels, exports, and visualizes.
    """
    df = pd.read_csv(input_csv)
    
    # 1. Generate text embeddings
    terms = df[term_column].tolist()
    text_embeddings = embed_texts(terms)
    
    # 2. Normalize numeric features
    metric_columns = ['impressions', 'clicks', 'conversions']
    # Ensure columns exist, fill missing with 0
    for col in metric_columns:
        if col not in df.columns:
            df[col] = 0
            
    scaler = MinMaxScaler()
    performance_metrics = scaler.fit_transform(df[metric_columns])
    
    # 3. Create hybrid embeddings by concatenating text and performance features
    hybrid_embeddings = pd.concat([pd.DataFrame(text_embeddings), pd.DataFrame(performance_metrics)], axis=1).values
    
    # 4. Cluster on hybrid embeddings
    if cluster_method == 'kmeans':
        labels = run_kmeans(hybrid_embeddings, n_clusters)
    elif cluster_method == 'hdbscan':
        labels = run_hdbscan(hybrid_embeddings)
    else:
        raise ValueError("Invalid cluster_method. Choose 'kmeans' or 'hdbscan'.")
        
    df["cluster_id"] = labels 
    
    # 5. Generate AI-powered labels and insights for each cluster
    ai_themes = {}
    ai_insights = {}
    unique_labels = sorted(list(set(labels)))

    for cluster_id in tqdm(unique_labels, desc="Generating AI Insights for Clusters"):
        if cluster_id == -1:
            ai_themes[cluster_id] = "Noise / Outliers"
            ai_insights[cluster_id] = "These terms did not fit into a specific cluster."
            continue

        cluster_terms = df[df['cluster_id'] == cluster_id][term_column].tolist()
        if cluster_terms:
            theme, insight = get_ai_cluster_analysis(cluster_terms)
            ai_themes[cluster_id] = theme
            ai_insights[cluster_id] = insight
        else:
            ai_themes[cluster_id] = f"Empty Cluster {cluster_id}"
            ai_insights[cluster_id] = "No terms found."
            
    # Map the AI analysis back to the main dataframe
    df['cluster_theme'] = df['cluster_id'].map(ai_themes)
    df['cluster_insight'] = df['cluster_id'].map(ai_insights)
    df = df.drop(columns=['cluster_label'], errors='ignore') # Remove old simple label

    df.to_csv(output_csv, index=False)
    
    # Visualize using the original text embeddings for semantic layout
    fig = plot_clusters(text_embeddings, labels)
    return fig

if __name__ == "__main__":
    main("data/processed/df_embeddings.csv", "output.csv", term_column='search_terms', cluster_method='kmeans', n_clusters=3)