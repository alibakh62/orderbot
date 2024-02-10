import warnings

warnings.filterwarnings("ignore")

############################## index ##############################
import os
import pandas as pd
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from sentence_transformers import SentenceTransformer

# Paths
VECTORDB_NAME_PREFIX = "db_0209"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # "" # Changed.
DATA_FILE_NAME = "grocery_products_data_desc.csv"  # Changed.
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
PRODUCT_VECTORDB_PATH = os.path.join(
    ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_product"
)
PRICE_VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_price")
HEALTH_VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_health")
PROMOTION_VECTORDB_PATH = os.path.join(
    ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_promotion"
)

# Read the product catalog data
product_catalog = pd.read_csv(os.path.join(ROOT_DIR, "data", DATA_FILE_NAME))


# Concatenate the columns
def concatenate_columns_product(row):
    return (
        str(row["Product_ID"])
        + "_"
        + str(row["Brand"])
        + " "
        + str(row["Product_Name"])
        + "_"
        + str(row["Category"])
        + "_"
        + str(row["Subcategory"])
    )


# Concatenate the columns # Not working.
def concatenate_columns_price(row):
    return (
        str(row["Product_ID"])
        + "_"
        + str(row["Brand"])
        + " "
        + str(row["Product_Name"])
        + "_"
        + str(row["Price"])
    )


# Concatenate the columns # Not working.
def concatenate_columns_health(row):
    return (
        str(row["Product_ID"])
        + "_"
        + str(row["Brand"])
        + " "
        + str(row["Product_Name"])
        + "_"
        + str(row["Nutrition"])
        + "_"
        + str(row["Ingredients"])
    )


# Concatenate the columns # Not working.
def concatenate_columns_promotion(row):
    return (
        str(row["Product_ID"])
        + "_"
        + str(row["Brand"])
        + " "
        + str(row["Product_Name"])
        + "_"
        + str(row["Promotions"])
    )


product_catalog["product_features"] = product_catalog.apply(
    concatenate_columns_product, axis=1
)

product_catalog["price_features"] = product_catalog.apply(
    concatenate_columns_price, axis=1
)

product_catalog["health_features"] = product_catalog.apply(
    concatenate_columns_health, axis=1
)

product_catalog["promotion_features"] = product_catalog.apply(
    concatenate_columns_promotion, axis=1
)


# Create the vectordb

# Download embeddings model
original_model = SentenceTransformer("all-MiniLM-L12-v2")
# Reload model using langchain wrapper
original_model.save(EMBEDDING_MODEL_PATH)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
# embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

feature_types = ["product", "price", "health", "promotion"]

for feature_type in feature_types:
    if feature_type == "product":
        documents = DataFrameLoader(
            product_catalog,
            page_content_column="product_features",
        ).load()
        VECTORDB_PATH = PRODUCT_VECTORDB_PATH
    elif feature_type == "price":
        documents = DataFrameLoader(
            product_catalog,
            page_content_column="price_features",
        ).load()
        VECTORDB_PATH = PRICE_VECTORDB_PATH
    elif feature_type == "health":
        documents = DataFrameLoader(
            product_catalog,
            page_content_column="health_features",
        ).load()
        VECTORDB_PATH = HEALTH_VECTORDB_PATH
    elif feature_type == "promotion":
        documents = DataFrameLoader(
            product_catalog,
            page_content_column="promotion_features",
        ).load()
        VECTORDB_PATH = PROMOTION_VECTORDB_PATH

    # Create the vector db
    vectordb = Chroma.from_documents(
        documents=documents, embedding=embedding_model, persist_directory=VECTORDB_PATH
    )
    vectordb.persist()


# Load the vector db
product_retriever = Chroma(
    persist_directory=PRODUCT_VECTORDB_PATH,
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 10})

# Load the vector db
price_retriever = Chroma(
    persist_directory=PRICE_VECTORDB_PATH,
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 10})

# Load the vector db
health_retriever = Chroma(
    persist_directory=HEALTH_VECTORDB_PATH,
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 10})

# Load the vector db
promotion_retriever = Chroma(
    persist_directory=PROMOTION_VECTORDB_PATH,
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 2})


############################## voice_bot_enhanced ##############################

import os
import json
import requests
import re

# from dotenv import load_dotenv # Changed.

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from openai import OpenAI
import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Paths
VECTORDB_NAME_PREFIX = "db_0209"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # "" # Changed.
DATA_FILE_NAME = "grocery_products_data_desc.csv"  # Changed.
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
PRODUCT_VECTORDB_PATH = os.path.join(
    ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_product"
)
PRICE_VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_price")
HEALTH_VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_health")
PROMOTION_VECTORDB_PATH = os.path.join(
    ROOT_DIR, "data", VECTORDB_NAME_PREFIX + "_promotion"
)

# Bot
# Setting up langsmith
import os
import time
from decimal import Decimal
from PIL import Image
import requests
from langsmith import Client

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

client = Client()


llm = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]


# The question is the last entry of the history
def extract_recent_chat(input, n=5):
    recents = input[-n:]
    context = ""
    for recent in recents:
        context += recent["content"] + " "
    return context


# The history is everything before the last question
def extract_history(input):
    return input[:-1]


# Customer intent detection chain
intent_detection_chain = (
    RunnablePassthrough()
    | {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
    }
    | PromptTemplate.from_template(
        """Your job is to identify the intent of the user. The user will be asking about a product and or a bunch of products. The user may want to know different things about the products. Given the user question below, classify the user's intent as 'HEALTH', 'PRICE', 'NUTRITION', 'INGREDIENTS', 'PROMOTIONS', 'PLACING_ORDER'.
    <question>
    {question}
    </question>
    Classification:"""
    )
    | llm
    | StrOutputParser()
)

# Price chain
price_chain = (
    RunnablePassthrough()
    | {"question": itemgetter("messages") | RunnableLambda(extract_recent_chat)}
    | {
        "question": itemgetter("question"),
        "context": itemgetter("question") | price_retriever,
    }
    | PromptTemplate.from_template(
        """Your job is to extract the price of the product from the price context data below inside triple backticks: \
        PRICE CONTEXT
        ```
        {context}
        ```
        The product price data has three features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, and the third feature is PRODUCT PRICE. Given the customer question below, identify the price of the product. If the product name does not exist in the product price data, return 'Product not found'. DO NOT make up any price. Only respond with product price.\
        <question>
        {question}
        </question>
        Price:"""
    )
    | llm
    | StrOutputParser()
)

# Promotion chain
promotion_chain = (
    RunnablePassthrough()
    | {"question": itemgetter("messages") | RunnableLambda(extract_recent_chat)}
    | {
        "question": itemgetter("question"),
        "context": itemgetter("question") | promotion_retriever,
    }
    | PromptTemplate.from_template(
        """Your job is to extract the promotion of the product from the promotion context data below inside triple backticks: \
        PROMOTION CONTEXT
        ```
        {context}
        ```
        The product promotion data has three features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, and the third feature is PRODUCT PROMOTION. Given the customer question below, identify whether there's a promotion running for a product or not. If the PRODUCT PROMOTION value is 'NA', it means there's no active promotion for this product. Any values other than that, means there is an active promotion. If there is an active promotion for a product, just return the exact value of PRODUCT PROMOTION. If there's no active promotion, respond with 'No promotion'. Only respond as instructed and do not make things up.\
        If there are multiple products in the promotion context, ALWAYS pick the product that has an active promotion. If there are multiple products with active promotions, pick the first one.\
        <question>
        {question}
        </question>
        Price:"""
    )
    | llm
    | StrOutputParser()
)

# Health chain
health_chain = (
    RunnablePassthrough()
    | {"question": itemgetter("messages") | RunnableLambda(extract_recent_chat)}
    | {
        "question": itemgetter("question"),
        "context": itemgetter("question") | health_retriever,
    }
    | PromptTemplate.from_template(
        """Your job is to extract the health information of the product from the health context data below inside triple backticks: \
        HEALTH CONTEXT
        ```
        {context}
        ```
        The product health data has three features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, the third feature is PRODUCT NUTRITION, and the fourth feature is PRODUCT INGREDIENTS. Given the customer question below, identify the health information of the product. If the product name does not exist in the product health data, return 'Product not found'. Only respond with product health information.\
        If the customer asks for a specific health information, respond with that information, otherwise, respond with only the nutritional values and the ingredient together and remove the '_' and speciy which part is nuritional value and which part is the ingredients.\
        For example, if the customer asks what's the calorie of Classic Potato Chips, respond with 'Calorie: 150'.\
        For example, if the customer asks if Classic Potato Chips is healthy, infer healthy based on the nutritional facts and ingredients and then show the actual values as well. Be short in your response.\
        <question>
        {question}
        </question>
        Health:"""
    )
    | llm
    | StrOutputParser()
)


question_with_history_and_context_str = """
You are OrderBot, an automated service to collect orders for a convenience store. You first collect the order, and then ask if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else. If it's a delivery, you ask for an address. \
Make sure to clarify all products, extras and sizes to uniquely identify the item from the product catalog. You respond in a short, very conversational friendly style. \
If the customer says something that is not clear, ask for clarification.\
Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".
Discussion: {chat_history}

The products present in the catalog are included below:\
PRODUCT CATALOG
{context}

The product context data has four features separated by '_'. The first feature is the PRODUCT ID, the second feature is the PRODUCT NAME, the third feature is PRODUCT CATEGORY, and the fourth feature is PRODUCT SUBCATEGORY. \
For example: Dole Organic Bananas_Produce_Fruit means Product_Name = 'Dole Organic Bananas' , Category = 'Produce', Subcategory = 'Fruit'.\
Ensure your responses are concise, factual, and designed to enhance the shopping experience by being responsive, informative, and patient. If a product is not available, mention its unavailability and suggest alternatives. Clarify all options, extras, and sizes to uniquely identify items from the product catalog.\
ALWAYS show products in the numbered list format and include the word "AVAILABLE OPTIONS" right before the numbered list. \
Example: If a customer wants to purchase RedBull, sandwich and hummus, the numbered list would be: 1. RedBull 2. Sandwich 3. Hummus\
Ask the customer to pick based on the numbers on the list. If the customer is not interested the product, show them other relevant products. Keep doing this until you show the customer all the available products. If the customer still can't find what she wanted, just apologize and say that we don't seem to have the item she's looking for.\
After the customer indicates their preference among the options provided, then inquire about the quantity they wish to order. At the same time, also suggest complementary products from the available product catalog that pair well with items in the customer's order but be subtle about it.\
For example, if a customer orders coffee, suggest adding a pastry or donuts to their order as a delightful complement. Continue to collect the entire order in this manner—offering selections, confirming choices, and then discussing quantities and prices.\

Once the order finalized, list the products that were selected by the user along with the quantities that the user selected and the total price of the item.
The following is an example format for the final prompt.
'''
Your order is finzlized! Here are your products.

SELECTED OPTIONS:

1. Green Tea, quantity: 1, price: $1.49
2. RedBull, quantity: 4, price: $27.24
3. Lays Chips, quantity: 12, price: $13.45
4. Brown bread, quantity: 13, price: $11.79
5. Organic bananas, quantity: 5, price: $2.80
If you need anything else in the future, feel free to reach out. Bye!
'''
Specifically format the final prompt in this manner. Once the order is finalized, include 'Bye!' in your final prompt.

Based on history, context, and the instructions provided above, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=question_with_history_and_context_str,
)

# def format_context(docs):
#     return "\n\n".join([d.page_content for d in docs])

order_chain = (
    RunnablePassthrough()
    | {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | {
        "context": itemgetter("question") | product_retriever,
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
    }
    | {"prompt": question_with_history_and_context_prompt}
    | {
        "result": itemgetter("prompt") | llm | StrOutputParser(),
    }
)

placeholder_chain = (
    RunnablePassthrough()
    | {
        "question": itemgetter("messages") | RunnableLambda(extract_recent_chat),
    }
    | PromptTemplate.from_template(
        """Respond to the following question:
        Question: {question}
        Answer:"""
    )
    | llm
    | StrOutputParser()
)

general_chain = (
    RunnablePassthrough()
    | {
        "question": itemgetter("messages") | RunnableLambda(extract_recent_chat),
    }
    | PromptTemplate.from_template(
        """Respond to the following question:
        Question: {question}
        Answer:"""
    )
    | llm
    | StrOutputParser()
)

branch = RunnableBranch(
    (lambda x: "price" in x["intent"].lower(), price_chain),
    (lambda x: "promotions" in x["intent"].lower(), promotion_chain),
    (lambda x: "health" in x["intent"].lower(), health_chain),
    (lambda x: "nutrition" in x["intent"].lower(), health_chain),
    (lambda x: "ingredients" in x["intent"].lower(), health_chain),
    (lambda x: "placing_order" in x["intent"].lower(), order_chain),
    general_chain,
)

# branch = RunnableBranch(
#     (lambda x: "price" in x["intent"].lower(), price_chain),
#     general_chain,
# )


full_chain = (
    RunnablePassthrough()
    | {
        "messages": lambda x: x["messages"],
        "intent": intent_detection_chain,
    }
    | branch
)


sample_msg_history = {
    "messages": [
        {"role": "user", "content": "Hey"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "I'd like a Potato Chips, please?"},
    ]
}

print(full_chain.invoke(sample_msg_history)["result"])

# chain = RunnablePassthrough() | {"messages": itemgetter("messages")}

# print(chain.invoke(sample_msg_history))
