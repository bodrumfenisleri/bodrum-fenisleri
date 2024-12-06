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

st.title("Bodrum 🤝 LangGraph")
st.markdown("#### Chat Streaming and Tool Calling from Bodrum turkey")

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

# Check if the OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    st.sidebar.header("OPENAI_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

# st write magic
with st.expander(label="Bodrum Chat Streaming and Tool Calling using LangGraph", expanded=st.session_state.expander_open):
    """
    You can query the Public Works Department's Netigma records in a database created using Pinecone vector store. 
    This advanced search system utilizes vector-based search technology with natural language processing capabilities to provide semantic search functionality. 
    Users can easily search through various documents including project files, technical reports, and administrative decisions using natural language queries. 
    The system processes these documents into vectors, enabling context-aware searches that go beyond traditional keyword matching. 
    Thanks to Pinecone's high-performance infrastructure, the system delivers fast and accurate results while maintaining scalability for large datasets.
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