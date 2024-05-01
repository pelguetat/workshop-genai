#!/usr/bin/env python
"""Example LangChain server exposes a retriever."""

# Import necessary modules for the server setup, language processing, and environmental variables.
from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from a specific file path for secure access to sensitive keys.
load_dotenv(
    "/Users/pabloelgueta/Documents/manning_course/qa-using-langchain-and-vector-databases-lp-author/project_3/.env"
)

# Retrieve API key from environment variables.
api_key = os.getenv("OPENAI_API_KEY")

# Define the model name for text embeddings.
model_name = "text-embedding-ada-002"

# Initialize OpenAIEmbeddings with the specific model, using the retrieved API key.
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

# Set up a vector storage system with specified parameters for embedding functions and persistence.
# Check that the path to your vector store directory is correct.
vectorstore = Chroma(
    collection_name="langchain_store",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Create a retriever instance using the previously configured vector store.
retriever = vectorstore.as_retriever()

# Set up a conversation memory buffer to store chat history, enabling message retrieval.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize a retrieval chain that connects language logic models (LLMs) with other components, specifying its type and integrations.
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(), chain_type="map_reduce", retriever=retriever, memory=memory
)

# Instantiate a FastAPI application with descriptive metadata.
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Register various application routes that allow interaction with the retriever.
add_routes(app, chain)

# Conditional check for direct script run, initializing server host and port settings.
if __name__ == "__main__":
    import uvicorn  # Import the ASGI server toolkit.

    # Launch the server on the specified host and port.
    uvicorn.run(app, host="localhost", port=8002)
