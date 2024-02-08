import warnings

warnings.filterwarnings("ignore")

import os
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
    return response.content, memory


def chatbot_interaction(user_input, conversation_history, memory):
    # response, updated_memory = get_answer(user_input, conversation_history)
    response, updated_memory = get_answer(user_input, memory)
    conversation_history.append((user_input, response))
    return conversation_history, response, updated_memory


with gr.Blocks() as demo:
    conversation_history = gr.State([])
    bot_memory = gr.State(memory)
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Enter your message here")

    user_input.submit(
        chatbot_interaction,
        inputs=[user_input, conversation_history, bot_memory],
        outputs=[chatbot, user_input, bot_memory],
    )

demo.launch()

# def speech_to_text(audio_data):
#     with open(audio_data, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1", response_format="text", file=audio_file
#         )
#     return transcript


# def text_to_speech(input_text):
#     # Initiate the request and get the response
#     response = client.audio.speech.create(model="tts-1", voice="nova", input=input_text)

#     audio_content = response.content

#     # Define the path for the output audio file
#     webm_file_path = "response_audio.mp3"

#     # Write the binary content to a file
#     with open(webm_file_path, "wb") as f:
#         f.write(audio_content)

#     return webm_file_path


# gr.ChatInterface(
#     get_answer,
#     # chatbot=gr.Chatbot(height=300),
#     # textbox=gr.Textbox(
#     #     placeholder="Ask me a yes or no question", container=False, scale=7
#     # ),
# ).launch()


# ----------------- Gradio -----------------


# ----------------- Streamlit -----------------
# def autoplay_audio(file_path: str):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode("utf-8")
#     md = f"""
#     <audio autoplay>
#     <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#     </audio>
#     """
#     st.markdown(md, unsafe_allow_html=True)


# # Float feature initialization
# # float_init()


# def initialize_session_state():
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Hi! How may I assist you today?"}
#         ]
#     if "audio_initialized" not in st.session_state:
#         st.session_state.audio_initialized = False
#     if "bot_memory" not in st.session_state:
#         st.session_state.bot_memory = memory


# initialize_session_state()

# st.title("AWS Hackathon AI Assistant 🤖")
# # audio = audiorecorder("Click to record", "Click to stop recording")

# # Create footer container for the microphone
# footer_container = st.container()

# with footer_container:
#     # cols = st.columns((1, 6))
#     audio_bytes = audio_recorder(
#         text="",
#         recording_color="#ff4500",
#         neutral_color="#1d3557",
#         icon_name="microphone",
#         icon_size="2x",
#     )

#     # cols[0].audiorecorder("Click to record", "Click to stop recording")
#     # cols[1].text_input("Start typing")

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "How may I assist you today?"}
#     ]
# else:
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

# if audio_bytes:
#     # Write the audio bytes to a file
#     with st.spinner("Transcribing..."):
#         webm_file_path = "temp_audio.mp3"
#         with open(webm_file_path, "wb") as f:
#             f.write(audio_bytes)

#         transcript = speech_to_text(webm_file_path)
#         if transcript:
#             st.session_state.messages.append({"role": "user", "content": transcript})
#             with st.chat_message("user"):
#                 st.write(transcript)
#             # os.remove(webm_file_path)

# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "How may I assist you today?"}
#     ]
# else:
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Interpreting 🤔..."):
#                 bot_memory = st.session_state.bot_memory
#                 final_response, bot_memory = get_answer(
#                     st.session_state.messages[-1]["content"], bot_memory
#                 )
#                 st.session_state.bot_memory = bot_memory
#                 print(f"User input: {st.session_state.messages[-1]['content']}")
#                 print(f"Final response: {final_response}")
#                 print(f"History: {bot_memory.load_memory_variables({})['history']}")
#                 print("-" * 100)
#             with st.spinner("Generating audio response 🎵..."):
#                 audio_file = text_to_speech(final_response)
#             if st.button("▶️"):
#                 autoplay_audio(audio_file)
#             st.write(final_response)
#             st.session_state.messages.append(
#                 {"role": "assistant", "content": final_response}
#             )
#             # os.remove(audio_file)

# # Float the footer container and provide CSS to target it with
# footer_container.float("bottom: 0rem;")
