import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from sentence_transformers import SentenceTransformer


# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_NAME = "walmart_grocery_sample.csv"
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
VECTORDB_PATH = os.path.join(ROOT_DIR, "data", "db")
print(f"ROOT_DIR: {ROOT_DIR}")

# Read the product catalog data
product_catalog = pd.read_csv(os.path.join(ROOT_DIR, "data", DATA_FILE_NAME))

# Sample the data
product_catalog_sample = (
    product_catalog[["index", "PRODUCT_NAME"]]
    .drop_duplicates(subset=["PRODUCT_NAME"])
    .copy()
    .head(100)
)
product_catalog_sample.rename(
    columns={"index": "id", "PRODUCT_NAME": "text"}, inplace=True
)
product_catalog_sample.to_csv(
    os.path.join(ROOT_DIR, "data", "product_catalog_sample.csv"), index=False
)
print(f"product_catalog: {product_catalog_sample.head()}")


# Create the vectordb

# Download embeddings model
original_model = SentenceTransformer("all-MiniLM-L12-v2")

# Reload model using langchain wrapper
original_model.save(EMBEDDING_MODEL_PATH)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

# Loading product dataframe
documents = DataFrameLoader(product_catalog_sample, page_content_column="text").load()

# Create the vector db
vectordb = Chroma.from_documents(
    documents=documents, embedding=embedding_model, persist_directory=VECTORDB_PATH
)

vectordb.persist()
