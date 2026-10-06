# loading environment variables
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# loading documents
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader  = PyPDFDirectoryLoader("./ai_papers")

pdf_documents = loader.load()

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)

pdf_chunks = splitter.split_documents(pdf_documents)

print(len(pdf_chunks)) 

# Embedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = AstraDBVectorStore(
    collection_name = "ragstck5oct2026_v2",
    embedding = embedding,
    api_endpoint = "https://777b60f7-b672-4f51-be31-e929b4f7d9de-us-east-2.apps.astra.datastax.com",
    token="AstraCS:xxxx:xxxx",

)

vector_store.add_documents(pdf_chunks)

# retriever

retriever = vector_store.as_retriever()

# prompt template

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# llm

llm = ChatGroq(model = "openai/gpt-oss-120b",
               temperature = 0.1)

# prompt

prompt = ChatPromptTemplate.from_template(
    ''' 
    You are useful AI assistant. Please answer user's query only on the basis of context available.

    context = {context}

    question : {input}

    '''
)

# create stuff chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

doc_chain = create_stuff_documents_chain(llm, prompt)

# retriever_chain

from langchain_classic.chains import create_retrieval_chain

retriever_chain = create_retrieval_chain(retriever, doc_chain)

# generate response

# response = retriever_chain.invoke({"input" : "What is transformer?"})

# print(response)

import streamlit as st

st.title("Chatbot with documents@AstraDB")

input_text  = st.text_input("Enter your query here")

if input_text:
    st.write(retriever_chain.invoke({'input' : input_text}))