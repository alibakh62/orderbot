import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain.schema import format_document
from operator import itemgetter
from langchain.memory import ConversationBufferMemory

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

# Test the retriever
# print([_.page_content for _ in retriever.get_relevant_documents("Hummus")])

# Bot
# Setting up langsmith
import os
from langsmith import Client

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

retriever = Chroma(
    persist_directory=VECTORDB_PATH,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH),
).as_retriever(search_kwargs={"k": 10})

system_template = """
You are OrderBot, an automated service to collect orders for a convenience grocery store. \
You first greet the customer, then collect the order, and then ask if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Make sure to clarify all options, extras and sizes to uniquely identify the item from the product catelog. \
You respond in a short, very conversational friendly style. \

The products is included below:

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

memory = ConversationBufferMemory(return_messages=True)
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

in_progress = True

while in_progress:
    memory.load_memory_variables({})
    user_input = input("User: ")
    inputs = {"input": user_input}
    response = chain.invoke(inputs)
    memory.save_context(inputs, {"output": f"{response.content}"})
    print(f"{response.content}")
    if "Bye!" in response.content:
        in_progress = False
