import os
import random
import asyncio
import streamlit as st
from pprint import pprint
from langchain_core.messages.tool import tool_call
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from astream_events_handler import invoke_our_graph
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
print(st.secrets["LANGCHAIN_PROJECT"])
st.set_page_config(page_title="Bodrum Belediye Başkanlığı Fen İşleri Müdürlüğü", page_icon="👷🏻")
st.title("Bodrum 🚧 Fen İşleri Müdürlüğü")
st.markdown("#### Netigma Yol Envanteri, Faaliyetler ve Ulakbel kayıtlarından oluşan veritabanı ile yapay zeka destekli sohbet")

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True



# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

# st write magic
with st.expander(label="Bodrum Chat Streaming and Tool Calling using LangGraph", expanded=st.session_state.expander_open):
    """
    Bu gelişmiş arama sistemi, anlamsal arama işlevselliği sağlamak için doğal dil işleme yetenekleriyle birlikte vektör tabanlı arama teknolojisini kullanır. 
    Kullanıcılar, doğal dil sorguları kullanarak faalyetler ve şikayetler gibi çeşitli belgelerde kolayca arama yapabilir. 
    Sistem, bu belgeleri vektörlere dönüştürerek, geleneksel anahtar kelime eşleştirmesinin ötesine geçen bağlam odaklı aramalar yapılmasını sağlar.

    """

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a placeholder container for streaming and any other events to visually render here
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))