import sys
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
import tempfile
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import TokenTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Import from langchain_ollama

assert sys.version_info >= (3, 10) and sys.version_info < (3, 13), "Use Python 3.10, 3.11, or 3.12 to run this notebook."

embeddings_model = OllamaEmbeddings(model="granite-embedding:30m-en")

model_name = "granite3.3:8b"
model = OllamaLLM(model=model_name)

query = "Who won in the Pantoja vs Asakura fight at UFC 310?"
prompt_template = PromptTemplate.from_template(template="{input}")

chain = prompt_template | model

output = chain.invoke({"input": query})
print(output)

query1 = "How much weight allowance is allowed in non championship fights in the UFC?"
output = chain.invoke({"input": query1})
print(output)

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
print(f"The vector database will be saved to {db_file}")

vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    enable_dynamic_field=True,
    index_params={"index_type": "AUTOINDEX"},
)

sources = [
    "https://www.ufc.com/news/main-card-results-highlights-winner-interviews-ufc-310-pantoja-vs-asakura",
    "https://media.ufc.tv/discover-ufc/Unified_Rules_MMA.pdf",
]

converter = DocumentConverter()
doc_id = 0

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

texts: list[Document] = []
for source in sources:
    doc = converter.convert(source=source).document
    
    # Extract text from the DoclingDocument
    # The error suggests using 'add_text' method or accessing text another way
    # Let's try extracting all text content from document paragraphs
    full_text = ""
    
    # Inspect document structure to see available content
    print(f"Document type: {type(doc)}")
    print(f"Document attributes: {dir(doc)}")
    
    # Try to get text from paragraphs if they exist
    if hasattr(doc, 'paragraphs'):
        for para in doc.paragraphs:
            if hasattr(para, 'text'):
                full_text += para.text + "\n\n"
    # If paragraphs don't exist, try to get text directly
    elif hasattr(doc, 'text'):
        full_text = doc.text
    # Last resort - try to convert to string
    else:
        full_text = str(doc)
    
    # Now split the full text into chunks
    chunks = text_splitter.split_text(full_text)
    
    for chunk in chunks:
        texts.append(Document(page_content=chunk, metadata={"doc_id": doc_id, "source": source}))
    doc_id += 1

print(f"{len(texts)} document chunks created")

# Only print the first few documents to avoid excessive output
for i, document in enumerate(texts):
    if i >= 5:  # Only show first 5 documents
        print(f"... and {len(texts) - 5} more document chunks")
        break
    print(f"Document ID: {document.metadata['doc_id']}")
    print(f"Source: {document.metadata['source']}")
    print(f"Content:\n{document.page_content}")
    print("=" * 80)

ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")

retriever = vector_db.as_retriever()
docs = retriever.invoke(query)
print(docs)

prompt = """
<|system|>You are a helpful assistant.  Use the context provided to answer the user's question.  If you don't know the answer, say you don't know.<|end_system|>
<|user|>
{input}
<|end_user|>
<|assistant|>
{context}
<|end_assistant|>
"""
prompt_template = PromptTemplate.from_template(template=prompt)

document_prompt_template = PromptTemplate.from_template(template="""<|start_of_role|>document {{\"document_id\": \"{doc_id}\"}}<|end_of_role|>{page_content}""")
document_separator = ""

combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt_template,
    document_prompt=document_prompt_template,
    document_separator=document_separator,
)
rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(),
    combine_docs_chain=combine_docs_chain,
)

output = rag_chain.invoke({"input": query})
print(output['answer'])

output = rag_chain.invoke({"input": query1})
print(output['answer'])