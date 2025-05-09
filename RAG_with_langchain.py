#!/usr/bin/env python3
# Retrieval Augmented Generation (RAG) with Langchain
# Using IBM Granite Models via Ollama

import os
import sys
import tempfile
import wget
import logging

# Set up logging to debug API issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_milvus import Milvus

# Check Python version
assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12 to run this program."

# Configure the models
OLLAMA_BASE_URL = "http://localhost:11434"  # Base URL without /api
EMBEDDING_MODEL = "granite-embedding:30m"
LLM_MODEL = "granite3.3:8b"

def main():
    print("Setting up RAG components...")
    
    # Initialize embedding model using Ollama
    try:
        embeddings_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        # Test the embedding model with a simple query
        print("Testing embedding model...")
        test_embedding = embeddings_model.embed_query("Test query")
        print(f"Embedding model test successful. Vector dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Make sure Ollama is running and the model is available.")
        print("You can pull the model with: ollama pull granite-embedding:30m")
        sys.exit(1)
    
    # Initialize LLM using Ollama
    try:
        llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        # Test the LLM with a simple query
        print("Testing LLM...")
        test_response = llm.invoke("Say hello")
        print(f"LLM test successful. Response: {test_response[:50]}...")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Make sure Ollama is running and the model is available.")
        print("You can pull the model with: ollama pull granite3.3:8b")
        sys.exit(1)
    
    # Set up vector database
    db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
    print(f"The vector database will be saved to {db_file}")
    
    vector_db = Milvus(
        embedding_function=embeddings_model,
        connection_args={"uri": db_file},
        auto_id=True,
        index_params={"index_type": "AUTOINDEX"},
    )
    
    # Download the document - State of the Union address
    filename = 'state_of_the_union.txt'
    url = 'https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'
    
    if not os.path.isfile(filename):
        print("Downloading sample document...")
        wget.download(url, out=filename)
    
    # Load and split the document
    print("\nSplitting document into chunks...")
    loader = TextLoader(filename)
    documents = loader.load()
    
    # Since we can't use transformers to get the tokenizer, we'll use a reasonable default chunk size
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    
    texts = text_splitter.split_documents(documents)
    doc_id = 0
    for text in texts:
        doc_id += 1
        text.metadata["doc_id"] = doc_id
    
    print(f"{len(texts)} text document chunks created")
    
    # Populate the vector database
    print("Populating vector database with document embeddings...")
    try:
        # Process documents in smaller batches to prevent overwhelming the API
        batch_size = 5
        all_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} documents)")
            batch_ids = vector_db.add_documents(batch_texts)
            all_ids.extend(batch_ids)
            
        print(f"{len(all_ids)} documents added to the vector database")
    except Exception as e:
        print(f"Error adding documents to vector database: {e}")
        sys.exit(1)
    
    # Test a similarity search
    query = "What did the president say about Ketanji Brown Jackson?"
    print(f"\nTest query: {query}")
    print("Running similarity search...")
    try:
        docs = vector_db.similarity_search(query)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        sys.exit(1)
    print(f"{len(docs)} documents returned")
    
    # Create RAG chain
    print("\nSetting up RAG pipeline...")
    
    # Create a prompt template for RAG
    prompt_template = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {input}

Answer:"""
    
    prompt = PromptTemplate.from_template(template=prompt_template)
    
    # Create a document prompt template
    document_prompt_template = PromptTemplate.from_template(template="Document {doc_id}:\n{page_content}")
    document_separator = "\n\n"
    
    # Assemble the RAG chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt_template,
        document_separator=document_separator,
    )
    
    rag_chain = create_retrieval_chain(
        retriever=vector_db.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    
    # Generate a response using RAG
    print("\nGenerating a response using RAG...")
    try:
        output = rag_chain.invoke({"input": query})
        print("\n--- RAG Answer ---")
        print(output['answer'])
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()