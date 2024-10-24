import streamlit as st
import nest_asyncio
import os
import pickle
import asyncio
from llama_index.llms.groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import QueryBundle
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Correct model import
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core import SummaryIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
embed_model = HuggingFaceEmbeddings()
#
Settings.embed_model = embed_model

# Initialize nest_asyncio to allow async operations in Streamlit
nest_asyncio.apply()

# Create and configure the general Groq LLM (Groq)
llm = Groq(
    model="llama3-70b-8192",  # or whatever model you wish to use for general processing
    api_key=os.getenv("GROQ_API_KEY")
)

# Set the API key as an environment variable for GROQ LLM
os.environ["GROQ_API_KEY"] = "API-KEY"
os.environ["PINECONE_API_KEY"] = "API-KEY"
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

documents = SimpleDirectoryReader(input_dir='corrected-english-circulars').load_data()
docs = []
for idx, doc in enumerate(documents):
    doc.metadata["index_id"] = str(doc.id_)
    docs.append(doc)
    
pinecone_index = pc.Index("english-circulars")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace="test",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

doc_index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context
)

async def aprocess_doc(doc, include_summary: bool = True):
    """Process doc."""
    metadata = doc.metadata

    # now extract out summary
    summary_index = SummaryIndex.from_documents([doc])
    query_str = "Give a detailed summary of this document consisting of 10-12 lines."
    query_engine = summary_index.as_query_engine(
        llm=Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    )
    
    # Introduce a delay to avoid rate limiting
    await asyncio.sleep(10)
    
    summary_txt = await query_engine.aquery(query_str)
    summary_txt = str(summary_txt)

    index_id = doc.metadata["index_id"]
    # filter for the specific doc id
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="index_id", operator=FilterOperator.EQ, value=str(index_id)
            ),
        ]
    )

    # create an index node using the summary text
    index_node = IndexNode(
        text=summary_txt,
        metadata=metadata,
        obj=doc_index.as_retriever(filters=filters),
        index_id=doc.id_,
    )

    return index_node


async def aprocess_docs(docs):
    """Process metadata on docs."""

    index_nodes = []
    tasks = []
    for doc in docs:
        task = aprocess_doc(doc)
        tasks.append(task)

    index_nodes = await run_jobs(tasks, show_progress=True, workers=3)

    return index_nodes

# Run the asynchronous document processing
index_nodes = await aprocess_docs(docs)

pinecone_index_metadata = pc.Index("english-circulars-metadata")
vector_store_auto = PineconeVectorStore(
    pinecone_index=pinecone_index_metadata,
    namespace="test2",
)
storage_context_auto = StorageContext.from_defaults(vector_store=vector_store_auto)
index = VectorStoreIndex(
    objects=index_nodes, storage_context=storage_context_auto
)
