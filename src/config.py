import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DATA_PATH = "./data/processed/"
EMBEDDING_PATH = "./data/embeddings/"


