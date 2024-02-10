import warnings

warnings.filterwarnings("ignore")

import os
import requests
import json
import re
from dotenv import load_dotenv

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
import gradio as gr


load_dotenv()

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = os.path.join(ROOT_DIR, "embedding_model")
VECTORDB_PATH = os.path.join(ROOT_DIR, "data", "db")


# Load the vector db
retriever = Chroma(
    persist_directory=VECTORDB_PATH,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH),
).as_retriever(search_kwargs={"k": 10})


# Bot
# Setting up langsmith
import os
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
You are OrderBot, an automated service to collect orders for a convenience grocery store. \
You first greet the customer, then collect the order, and then ask if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Make sure to clarify all options, extras and sizes to uniquely identify the item from the product catelog. \
If the number of options for an item is more than three, select three options and ask the user to select one. Make sure to show the options in a numbered list format. Ask the customer to pick based on the numbers on the list. If the user can't find the item she wanted, show her another five options. Keep doing this until you show the customer all the available items. If the customer still can't find what she wanted, just apologize and say that we don't seem to have the item she's looking for.\
ALWAYS show product name options in the numbered list format and include the word "OPTIONS" right before the numbered list. \
You respond in a short, very conversational friendly style. \
If the customer says something that is not clear, ask for clarification. \
You only want to offer products from the product catalog listed below. If the customer's oder contain a product that is not in the product catalog, just apologize and say we don't have it. \

The products is included below:

PRODUCT CATALOG
{product_catalog}

Once the order finalized, include 'Bye!' in your final prompt."""

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

global memory
memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  # Replace with your own key or find a way to use it indirectly.
webm_file_path = "response_audio.mp3"


# ----------------- Helpers -----------------
# Card component helpers
def extract_product_names(input_string):
    # Check if "OPTIONS" is in the input string
    if "OPTIONS" not in input_string:
        return "The input string does not contain product options."

    # List to hold extracted product names
    product_names = []

    # Split the string into lines
    lines = input_string.split("\n")

    # Regular expression to match product names
    # It looks for lines starting with a digit (the option number) followed by a period and a space, then captures the rest of the line
    pattern = re.compile(r"^\d+\.\s+(.*)")

    for line in lines:
        # Try to find a match for the pattern
        match = pattern.match(line)
        if match:
            # If a match is found, add the captured group (the product name) to the list
            product_names.append(match.group(1))

    return product_names


def search(query):
    url = "https://google.serper.dev/images"

    payload = json.dumps({"q": query, "num": 3})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = response.json()
    # print(f"Search result for {query}: {result}")
    image_url = result["images"][0]["imageUrl"]
    filename = os.path.join(ROOT_DIR, "images", query.replace(" ", "_") + ".jpg")

    def download_image(url):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                file.write(response.content)
                print(f"Image successfully downloaded: {filename}")
        else:
            print(f"HTTP request failed with status code {response.status_code}")

    download_image(image_url)
    return filename


# ----------------- Gradio -----------------


def get_answer(message, memory):
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
    inputs = {"input": message}
    response = chain.invoke(inputs)
    memory.save_context(inputs, {"output": f"{response.content}"})
    print(memory.load_memory_variables({})["history"])
    return response.content, memory


# def chatbot_interaction(user_input, conversation_history, memory):
#     response, updated_memory = get_answer(user_input, memory)
#     conversation_history.append((user_input, response))
#     return conversation_history, response  # , updated_memory


def chatbot_interaction(user_input, conversation_history, memory):
    response, updated_memory = get_answer(user_input, memory)
    conversation_history.append((user_input, response))

    # Logic for Image Display
    filenames = []
    if "OPTIONS" in response:
        product_names = extract_product_names(response)
        print(f"Product names: {product_names}")
        for product_name in product_names:
            filenames.append(search(product_name))
        print(f"Image filenames: {filenames}")
        # Create a Gradio row to display images
        # image_row = gr.Row()
        # for filename in filenames:
        #     image_row.add(gr.Image(filename))

        # Return chatbot history, text response, AND image row
        return conversation_history, response, filenames  # image_row

    else:
        # Create an empty row to maintain output consistency
        # image_row = gr.Row()
        # Return chatbot history and text response (no images)
        return conversation_history, response, filenames  # image_row


with gr.Blocks() as demo:
    conversation_history = gr.State([])
    image_files = gr.State([])
    bot_memory = gr.State(memory)
    response = gr.State("")
    with gr.Row():
        chatbot = gr.Chatbot(bubble_full_width=False, scale=2)
        with gr.Column(scale=1):
            gr.Text(image_files.value)
            # gr.Image(image_files[0], label="Product Image")
            # gr.Image(image_files[1], label="Product Image")
            # gr.Image(image_files[2], label="Product Image")

    user_input = gr.Textbox(placeholder="Enter your message here")

    user_input.submit(
        chatbot_interaction,
        inputs=[user_input, conversation_history, bot_memory],
        outputs=[chatbot, response, image_files],
    )
    print(image_files)

# with gr.Blocks() as demo:
#     conversation_history = gr.State([])
#     bot_memory = gr.State(memory)
#     response = gr.State("")
#     chatbot = gr.Chatbot()
#     user_input = gr.Textbox(placeholder="Enter your message here")

#     user_input.submit(
#         chatbot_interaction,
#         inputs=[user_input, conversation_history, bot_memory],
#         outputs=[chatbot, response],
#     )
#     print(response)

demo.launch()
