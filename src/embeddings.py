import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY


def embed_texts(text_list, model=EMBEDDING_MODEL):
    embeddings = []
    for text in tqdm(text_list):
        try:
            response = openai.embeddings.create(input=text, model=model)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Embedding failed for '{text}': {e}")
            embeddings.append([0] * 1536)
    return np.array(embeddings)


def embed_dataframe_column(df, column_name, output_path):
    unique_texts = df[column_name].dropna().unique().tolist()
    vectors = embed_texts(unique_texts)
    emb_df = pd.DataFrame(vectors, index=unique_texts)
    emb_df.to_csv(output_path)
    return emb_df
