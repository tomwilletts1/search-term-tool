import streamlit as st
import pandas as pd
import os
import openai
import plotly.express as px
from src.modeling.train import main as run_pipeline
from src.services.openai_services import get_ai_cluster_analysis, get_contextual_chat_response

# --- Helper Functions ---
def calculate_cluster_metrics(df):
    """Calculates and aggregates performance metrics for each cluster."""
    if df.empty or 'cluster_id' not in df.columns:
        return pd.DataFrame()
    agg_metrics = {
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'search_term': 'count'
    }
    cluster_summary = df.groupby(['cluster_id', 'cluster_theme', 'cluster_insight']).agg(agg_metrics).reset_index()
    cluster_summary = cluster_summary.rename(columns={'search_term': 'num_terms'})
    cluster_summary['ctr'] = (cluster_summary['clicks'] / cluster_summary['impressions']).fillna(0)
    cluster_summary['conversion_rate'] = (cluster_summary['conversions'] / cluster_summary['clicks']).fillna(0)
    cluster_summary['impressions'] = cluster_summary['impressions'].apply(lambda x: f"{int(x):,}")
    cluster_summary['clicks'] = cluster_summary['clicks'].apply(lambda x: f"{int(x):,}")
    cluster_summary['conversions'] = cluster_summary['conversions'].apply(lambda x: f"{int(x):,}")
    cluster_summary['ctr'] = cluster_summary['ctr'].apply(lambda x: f"{x:.2%}")
    cluster_summary['conversion_rate'] = cluster_summary['conversion_rate'].apply(lambda x: f"{x:.2%}")
    return cluster_summary

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Search Term Clustering Tool")

# --- Sidebar: All controls, API key, file uploader, and chat ---
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        st.session_state['api_key'] = api_key
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # Ensure env var is set for all libraries
    elif 'api_key' in st.session_state:
        openai.api_key = st.session_state['api_key']
        os.environ["OPENAI_API_KEY"] = st.session_state['api_key']
    else:
        openai.api_key = None  # No key set
        os.environ.pop("OPENAI_API_KEY", None)  # Remove env var if not set
    if not openai.api_key:
        st.warning("Please enter your OpenAI API key to use AI features.")

    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="sidebar_uploader")
    term_column = None
    cluster_method = None
    n_clusters = 3
    if uploaded_file:
        if 'full_df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.full_df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.session_state.file_name = uploaded_file.name
            st.session_state.ran_clustering = False
        df_preview = st.session_state.full_df
        term_column = st.selectbox("Select the column with search terms", df_preview.columns)
        st.subheader("Clustering Settings")
        cluster_method = st.selectbox("Clustering Method", ["kmeans", "hdbscan"])
        if cluster_method == "kmeans":
            n_clusters = st.slider("Number of Clusters (KMeans only)", 2, 20, 5)
        if st.button("Run Clustering & Analysis", type="primary"):
            with st.spinner("Processing... This may take a few minutes for AI analysis."):
                # No longer save uploaded file to data/processed
                # Instead, process directly in-memory
                input_csv_path = "input_temp.csv"
                output_csv_path = "output.csv"
                st.session_state.full_df.to_csv(input_csv_path, index=False)
                fig = run_pipeline(input_csv_path, output_csv_path, term_column=term_column, cluster_method=cluster_method, n_clusters=n_clusters)
                st.session_state.results_df = pd.read_csv(output_csv_path)
                st.session_state.summary_df = calculate_cluster_metrics(st.session_state.results_df)
                st.session_state.plot_fig = fig
                st.session_state.ran_clustering = True
            st.success("Analysis Complete!")

    st.header("💬 Chat with your Data")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask about your clusters..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                context_summary = st.session_state.summary_df.to_string() if 'summary_df' in st.session_state else ""
                response = get_contextual_chat_response(prompt, context_summary)
                message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App UI ---
st.title("🧠 AI-Powered Search Term Clustering Tool")
st.markdown("Upload your search query data to group terms by semantic meaning and performance, then get AI-powered insights.")

if st.session_state.get('ran_clustering', False):
    st.header("📊 Cluster Analysis Dashboard")
    metric_options = ['impressions', 'clicks', 'conversions']
    metric_labels = {'impressions': 'Impressions', 'clicks': 'Clicks', 'conversions': 'Conversions'}
    selected_metric = st.selectbox("Select metric for bar chart", options=metric_options, format_func=lambda x: metric_labels[x])
    st.subheader(f"{metric_labels[selected_metric]} by Cluster (Bar Chart)")
    bar_df = st.session_state.summary_df.copy()
    bar_df[selected_metric] = bar_df[selected_metric].str.replace(',', '').astype(int)
    fig_bar = px.bar(bar_df, x='cluster_theme', y=selected_metric, color='cluster_theme',
                    title=f'Total {metric_labels[selected_metric]} by Cluster', labels={selected_metric: metric_labels[selected_metric], 'cluster_theme': 'Cluster Theme'})
    st.plotly_chart(fig_bar, use_container_width=True)
    cluster_options = ['All Clusters'] + sorted(st.session_state.summary_df['cluster_theme'].unique().tolist())
    selected_cluster_theme = st.selectbox("Filter by Cluster Theme", options=cluster_options)
    st.subheader("Cluster Performance Summary")
    display_summary = st.session_state.summary_df
    if selected_cluster_theme != 'All Clusters':
        display_summary = display_summary[display_summary['cluster_theme'] == selected_cluster_theme]
    st.dataframe(display_summary, use_container_width=True)
    st.subheader("Search Term Details")
    display_details = st.session_state.results_df
    if selected_cluster_theme != 'All Clusters':
        selected_cluster_id = st.session_state.summary_df[st.session_state.summary_df['cluster_theme'] == selected_cluster_theme]['cluster_id'].iloc[0]
        display_details = display_details[display_details['cluster_id'] == selected_cluster_id]
    st.dataframe(display_details)
else:
    st.info("Upload a file and run the clustering analysis to see your results.")
