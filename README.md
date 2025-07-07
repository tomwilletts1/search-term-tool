# 🧠 AI-Powered Search Term Clustering Tool

This app allows you to upload a CSV of search terms and performance metrics (impressions, clicks, conversions). It then generates hybrid embeddings to cluster terms based on both **semantic meaning** and **performance characteristics**.

The tool uses the OpenAI GPT-4o model to provide intelligent cluster labels, marketing insights, and a live chat interface to ask questions about your data.

---

## Features

- **Hybrid Clustering**: Combines text embeddings with performance metrics for more commercially relevant clusters.
- **AI-Powered Labeling**: Each cluster is automatically given a descriptive theme and an actionable marketing insight by GPT-4o.
- **Interactive Dashboard**: View a high-level summary of all clusters, including aggregated performance metrics.
- **Cluster Filtering**: Drill down into specific clusters to analyze their performance and the terms within them.
- **Live Chat Analysis**: Use the sidebar to ask questions about your clusters (e.g., "Why is Cluster 3 underperforming?") and get contextual answers from an AI analyst.

---

## Project Structure

```
Search Term Tool/
│
├── app.py                      # Streamlit app entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   └── processed/              # Caching for processed data
│
├── src/
│   ├── config.py               # API keys and config
│   ├── embeddings.py           # Text embedding generation
│   ├── modeling/
│   │   └── train.py            # Core data processing pipeline
│   ├── services/
│   │   └── openai_services.py  # AI labeling and chat logic
│   ├── clustering/
│   │   ├── kmeans.py
│   │   ├── hdbscan.py
│   │   └── labeling.py         # (Legacy simple labeling)
│   └── visualization/
│       └── plot_clusters.py    # UMAP + matplotlib plotting
```

---

## Setup and Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set OpenAI API Key**: Your key is read from your environment variables.
    *   **PowerShell**: `$env:OPENAI_API_KEY="your-secret-api-key"`
    *   **Bash/Zsh**: `export OPENAI_API_KEY="your-secret-api-key"`

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

4.  **Using the Tool**:
    - Upload your CSV file via the sidebar. It should contain a column for search terms and can optionally include `impressions`, `clicks`, and `conversions`.
    - Select the correct search term column.
    - Adjust clustering settings if desired.
    - Click **"Run Clustering & Analysis"**.
    - Explore the dashboard, filter by cluster, and use the chat to ask for deeper insights.

---

## Notes
- All output files are saved in the `data/processed/` directory.
- The app uses UMAP for 2D visualization and matplotlib for plotting.
- The codebase is modular and easy to extend for new clustering or embedding methods.
