import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.pdf import PyMuPDFLoader
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
def process_pdf(uploaded_file):
    tmp_location = os.path.join('', uploaded_file.name)
    with open(tmp_location, 'wb') as f:
        f.write(uploaded_file.getvalue())

    loader = PyMuPDFLoader(file_path=tmp_location)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 1.0, "max_length": 256})
    global chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return 'Document has successfully been loaded'

st.title("PDF Document Processor")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    result = process_pdf(uploaded_file)
    st.success(result)
    query = st.text_input("Enter your query:")
    if query:
        response = chain.run(query)
        st.write(response)