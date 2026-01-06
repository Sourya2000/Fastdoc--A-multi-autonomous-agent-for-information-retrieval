from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Dict, Optional, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# -------------------
# LLM + Embeddings
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# -------------------
# Thread Stores
# -------------------
_THREAD_RETRIEVERS: Dict[str, any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(str(thread_id))


# -------------------
# PDF Ingestion
# -------------------
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        os.remove(tmp_path)


# -------------------
# Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed"}

    docs = retriever.invoke(query)

    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "metadata": [d.metadata for d in docs],
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }


tools = [search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = config["configurable"]["thread_id"]

    system = SystemMessage(
        content=(
            "Answer user questions. "
            "If question is about the PDF, call rag_tool "
            f"with thread_id={thread_id}"
        )
    )

    response = llm_with_tools.invoke(
        [system, *state["messages"]],
        config=config,
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# -------------------
# Checkpointer
# -------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# Helpers
# -------------------
def retrieve_all_threads():
    return [
        c.config["configurable"]["thread_id"]
        for c in checkpointer.list(None)
    ]

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
