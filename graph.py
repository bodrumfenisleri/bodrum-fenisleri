import os
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import model
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse
from langchain_qdrant import SparseEmbeddings
from langchain.tools import StructuredTool
from typing import List
from langchain.schema import Document
#from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_core.messages import SystemMessage
load_dotenv()

os.environ["QDRANT_API_KEY_EBARTAN"] = st.secrets["QDRANT_API_KEY_EBARTAN"]
os.environ["QDRANT_URL_EBARTAN"] = st.secrets["QDRANT_URL_EBARTAN"]
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


index_name = "bodrum"

pinecone_vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name,namespace="netigma")
pinecone_vector_store_ulakbel = PineconeVectorStore(embedding=embeddings, index_name=index_name,namespace="ulakbel")

retriever_pinecone = pinecone_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.9},
)
retriever_pinecone_ulakbel = pinecone_vector_store_ulakbel.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.9},
)


def search_pinecone(query: str) -> List[Document]:
    """
    Perform a search using Pinecone vector store for netigma namespace
    
    Args:
        query (str): The search pinecone as_retriever to be used
        
    Returns:
        List[Document]: List of retrieved documents
    """
    return retriever_pinecone.invoke(query)

def search_pinecone_uakbel(query: str) -> List[Document]:
    """
    Perform a search using Pinecone vector store for ulakbel namespace
    
    Args:
        query (str): The search pinecone as_retriever to be used
        
    Returns:
        List[Document]: List of retrieved documents
    """
    return retriever_pinecone_ulakbel.invoke(query)
search_PINECONE = StructuredTool.from_function(
        name="PineconeSearch",
        func=search_pinecone,  # Executes Pinecone search using the provided query
        description=f"""
        A mandatory vector store search tool that retrieves information from the Netigma namespace.
        This tool MUST be used in every search operation.
        Input should be a search query string.
        """,
    )
search_PINECONE_ULAKBEL = StructuredTool.from_function(
        name="PineconeSearchULAKBEL",
        func=search_pinecone_uakbel,  # Executes Pinecone search using the provided query
        description=f"""
        A mandatory vector store search tool that retrieves information from the Ulakbel namespace.
        This tool MUST be used in every search operation.
        Input should be a search query string.
        """,
    )

# List of tools that will be accessible to the graph via the ToolNode
tools = [search_PINECONE,search_PINECONE_ULAKBEL]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed


def _call_model(state: GraphsState):
    
    messages = state["messages"]
    system_message = {
    "role": "system",
    "content": """You MUST use both 'PineconeSearch' and 'PineconeSearchULAKBEL' tools for every search query.
    Combine information from both sources to create a comprehensive response.
    Do not provide an answer without using both tools."""}
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=system_message["content"])] + messages
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
    ).bind_tools(tools, parallel_tool_calls=True)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()