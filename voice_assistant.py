############################## index ##############################
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from sentence_transformers import SentenceTransformer

# Paths
VECTORDB_NAME = "db_0209"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # "" # Changed.
DATA_FILE_NAME = "grocery_products_data_desc.csv"  # Changed.
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME)

# Read the product catalog data
product_catalog = pd.read_csv(os.path.join(ROOT_DIR, "data", DATA_FILE_NAME))


# Concatenate the columns # Not working.
def concatenate_columns(row):
    return (
        str(row["Brand"])
        + " "
        + str(row["Product_Name"])
        + "_"
        + str(row["Category"])
        + "_"
        + str(row["Subcategory"])
        + "_"
        + str(row["Order_Unit"])
        + "_"
        + str(row["Price"])
        + "_"
        + str(row["Nutrition"])
        + "_"
        + str(row["Ingredients"])
    )


product_catalog["features"] = product_catalog.apply(
    concatenate_columns, axis=1
)  # Apply the function to each row of the DataFrame


# Create the vectordb

# Download embeddings model
original_model = SentenceTransformer("all-MiniLM-L12-v2")
# Reload model using langchain wrapper
original_model.save(EMBEDDING_MODEL_PATH)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
# embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Loading product dataframe
documents = DataFrameLoader(product_catalog, page_content_column="features").load()

# Create the vector db
vectordb = Chroma.from_documents(
    documents=documents, embedding=embedding_model, persist_directory=VECTORDB_PATH
)

vectordb.persist()

# Load the vector db
retriever = Chroma(
    persist_directory=VECTORDB_PATH,
    embedding_function=embedding_model,
).as_retriever(search_kwargs={"k": 10})

############################## voice_bot_enhanced ##############################
import warnings

warnings.filterwarnings("ignore")

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
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from operator import itemgetter
from langchain.memory import ConversationBufferMemory

from openai import OpenAI
import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # ""  # Changed.
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
VECTORDB_PATH = os.path.join(ROOT_DIR, "data", VECTORDB_NAME)

# Test the retriever
# print([_.page_content for _ in retriever.get_relevant_documents("Hummus")])

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
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


system_template = """
You are OrderBot, an automated service to collect orders for a convenience store. You first collect the order, and then ask if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else. If it's a delivery, you ask for an address. \
Make sure to clarify all products, extras and sizes to uniquely identify the item from the product catalog. You respond in a short, very conversational friendly style. \
If the customer says something that is not clear, ask for clarification.\
The products present in the catalog are included below:\
{product_catalog}
Understand the products that the user needs and generate relevant options for each product that the customer desires using the produt catalog.\
In the product catalog, the product name is everything before the first '_'. The features are separated by '_' and these features include Product_Name, Category, Subcategory, Order_Unit, Price, Nutritional_Facts, and Ingredients.\
For example: Dole Organic Bananas_Produce_Fruit_lbs_$0.59_89 calories, 0.3g fat, 1.1g protein_Bananas means Product_Name = 'Dole Organic Bananas' , Category = 'Produce', Subcategory = 'Fruit_lbs', Order_Unit = 'lbs' Price = '$0.59', Nutritional_Facts = '0.59_89 calories, 0.3g fat, 1.1g protein', Ingredients = 'Bananas'\
Ensure your responses are concise, factual, and designed to enhance the shopping experience by being responsive, informative, and patient. If a product is not available, mention its unavailability and suggest alternatives. Clarify all options, extras, and sizes to uniquely identify items from the product catalog.\
ALWAYS show products in the numbered list format and include the word " AVAILABLE OPTIONS" right before the numbered list. \ Example: If a customer wants to purchase RedBull, sandwich and hummus, the numbered list would be: 1. RedBull 2. Sandwich 3. Hummus\
Ask the customer to pick based on the numbers on the list. If the customer is not interested the product, show them other relevant products. Keep doing this until you show the customer all the available products. If the customer still can't find what she wanted, just apologize and say that we don't seem to have the item she's looking for.\
After the customer indicates their preference among the options provided, then inquire about the quantity they wish to order. At the same time, also suggest complementary products from the available product catalog that pair well with items in the customer's order but be subtle about it.\
For example, if a customer orders coffee, suggest adding a pastry or donuts to their order as a delightful complement. Continue to collect the entire order in this manner—offering selections, confirming choices, and then discussing quantities and prices.\
If customer order involves some health attribute of the product instead of mentioning the product name, use the Nutritional_Facts and Ingredients data from the product catalog to provide the customer with the information they need. For example, if the customer asks: 'Give me a few healthy snacks option.', you should infer healthy from Nutritional_Facts and Ingredients data.\

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
Specifically formatnthe final prompt in this manner. Once the order is finalized, include 'Bye!' in your final prompt."""

user_template = "User: {input}"

system_prompt = SystemMessagePromptTemplate.from_template(system_template)
user_prompt = HumanMessagePromptTemplate.from_template(user_template)

prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        MessagesPlaceholder(variable_name="history"),
        user_prompt,
    ]
)

memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})

client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY") # Changed.
    api_key=os.getenv("OPENAI_API_KEY")
)  # Replace with your own key or find a way to use it indirectly.
webm_file_path = "response_audio.mp3"

# ----------------------- Helpers -----------------------


# Voice helpers
def get_answer(user_input, memory):
    memory.load_memory_variables({})
    loaded_memory = RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    chain = (
        loaded_memory
        | {
            "input": lambda x: x["input"],
            "product_catalog": itemgetter("input") | retriever,
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
    )
    inputs = {"input": user_input}
    response = chain.invoke(inputs)
    memory.save_context(inputs, {"output": f"{response.content}"})
    return response.content, memory


def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", response_format="text", file=audio_file
        )
    return transcript


def text_to_speech(input_text):
    # Initiate the request and get the response
    response = client.audio.speech.create(model="tts-1", voice="nova", input=input_text)

    audio_content = response.content

    # Define the path for the output audio file
    # webm_file_path = "response_audio.mp3"

    # Write the binary content to a file
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)

    return webm_file_path


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


# Card component helpers
def extract_product_names(input_string):
    # Check if "OPTIONS" is in the input string
    if "OPTIONS" not in input_string:
        return "The input string does not contain product options."

    # List to hold extracted product names, quantities and prices
    product_names = []
    quantities = []
    prices = []

    # Split the string into lines
    lines = input_string.split("\n")

    # Regular expression to match product names
    # It looks for
    pattern = re.compile(
        r"^\d+\.\s+([^,]*)"
    )  # Lines starting with a digit followed by a period and a space; captures the line until a comma is encountered.
    quantity = re.compile(
        r"^\d+\.\s.*quantity:\s([^,]+),"
    )  # Lines starting with a digit followed by a period and a space; captures from 'quantity:' to ','
    price = re.compile(
        r"^\d+\.\s+.*?price: \$([^ ]+)"
    )  # Lines starting with a digit followed by a period and a space; captures from 'price: $' to ' ' or end of the line.
    for line in lines:
        # Try to find a match for the pattern
        match1 = pattern.match(line)
        match2 = quantity.match(line)
        match3 = price.match(line)
        if match1:
            # If a match is found, add the captured group (the product name) to the list
            product_names.append(match1.group(1))
        if match2:
            # If a match is found, add the captured group (the quantity) to the list
            quantities.append(match2.group(1))
        if match3:
            # If a match is found, add the captured group (the price) to the list
            prices.append(match3.group(1))

    return product_names, quantities, prices


def search(query):
    url = "https://google.serper.dev/images"

    payload = json.dumps({"q": query, "num": 3})
    headers = {
        # "X-API-KEY": os.getenv("SERPER_API_KEY"), # Changed.
        "X-API-KEY": "df7d4e1e23ec76ea9ef50282a9a24d5a72a392e8",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = response.json()
    image_url = result["images"][0]["imageUrl"]
    return image_url


# ----------------------- Streamlit -----------------------

# Float feature initialization
float_init()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! What would you like to order today?",
            }
        ]
    if "audio_initialized" not in st.session_state:
        st.session_state.audio_initialized = False
    if "bot_memory" not in st.session_state:
        st.session_state.bot_memory = memory


initialize_session_state()

st.title("AWS Hackathon AI Assistant 🤖")
# audio = audiorecorder("Click to record", "Click to stop recording")

# Create footer container for the microphone
footer_container = st.container()

with footer_container:
    audio_bytes = audio_recorder(
        text=None,
        recording_color="#ff4500",
        neutral_color="#1d3557",
        icon_name="microphone",
        icon_size="2x",
    )

    # cols[0].audiorecorder("Click to record", "Click to stop recording")
    # cols[1].text_input("Start typing")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! What would you like to order today?"}
    ]
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing🖋️..."):
        webm_file_path = "input_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
        transcript = None
    # audio_bytes = None

# Create input prompt.
if text_prompt := st.chat_input("Type something or use the microphone to record"):
    st.session_state.messages.append({"role": "user", "content": text_prompt})
    with st.chat_message("user"):
        st.write(text_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! What would you like to order today?"}
    ]
else:
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Interpreting 🤔..."):
                bot_memory = st.session_state.bot_memory
                final_response, bot_memory = get_answer(
                    st.session_state.messages[-1]["content"], bot_memory
                )
                st.session_state.bot_memory = bot_memory
            with st.spinner("Generating audio response 🎵..."):
                audio_file = text_to_speech(final_response)
                with open(webm_file_path, "rb") as f:
                    data = f.read()
                    st.audio(data, format="audio/mp3")
            if "OPTIONS" in final_response:
                st.write(final_response)
                # Extract product names, quantities and prices from the response
                product_names, quantities, prices = extract_product_names(
                    final_response
                )
                # Get the product images
                filenames = []
                for product_name in product_names:
                    filenames.append(search(product_name))
                # Display the product images
                with st.container():
                    cols = st.columns(len(filenames))
                    for i, filename in enumerate(filenames):
                        raw_image = Image.open(requests.get(filename, stream=True).raw)
                        resized_image = raw_image.resize((600, 600))
                        tile = cols[i].container()
                        tile.image(
                            resized_image,
                            caption=f"{i+1}. {product_names[i]}",
                            width=150,
                        )
            else:
                st.write(final_response)

            if "bye!" in final_response.lower():
                st.toast("Thank you for chatting with our AI!")
                time.sleep(1)
                st.toast("Your order has been finalized, preparing your cart...")
                time.sleep(1)
                st.toast("Hooray! Your cart is ready for checkout!", icon="🙂")

                with st.sidebar:
                    st.markdown(
                        "<h1 style='text-align: center; color: #1d3557;'>Your Cart 🛒</h1>",
                        unsafe_allow_html=True,
                    )

                    for i, filename in enumerate(filenames):
                        with st.container(border=True):
                            st.markdown(
                                f"<h6 style='text-align: center; color: #1d3557;'>{i+1}. {product_names[i]}</h6>",
                                unsafe_allow_html=True,
                            )
                            raw_image = Image.open(
                                requests.get(filename, stream=True).raw
                            )
                            resized_image = raw_image.resize((1024, 1024))
                            tile = st.columns([2, 1.5, 1])
                            tile[0].image(resized_image, width=100)
                            tile[1].write(f"Quantity: {quantities[i]}")
                            tile[2].write(f"Price: ${prices[i]}")
                    st.divider()
                    product_names, quantities, prices = extract_product_names(
                        final_response
                    )
                    # subtotal_list = [float(prices[i]) for i in range(0, len(prices))]
                    subtotal_list = [Decimal(ele) for ele in prices]
                    subtotal = sum(subtotal_list)
                    taxes = round(Decimal("0.06") * subtotal, 2)
                    delivery_fee, service_fee = Decimal("1.59"), Decimal("2.49")
                    total = round(subtotal + taxes + delivery_fee + service_fee, 2)
                    checkout_df = pd.DataFrame(
                        {
                            "Expenses": [
                                "Subtotal",
                                "Taxes",
                                "Delivery fee",
                                "Service fee",
                            ],
                            "Amount": [subtotal, taxes, delivery_fee, service_fee],
                        }
                    )
                    with st.container(border=False):
                        st.markdown(
                            f"<h1 style='text-align: center; color: #1d3557;'>Your order total is: ${total}</h1>",
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"Full details"):
                            st.dataframe(checkout_df, hide_index=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

# Float the footer container and provide CSS.
footer_container.float("bottom: 3rem; left: 72rem")
