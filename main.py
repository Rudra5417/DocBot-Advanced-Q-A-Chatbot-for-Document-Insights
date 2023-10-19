import streamlit as st
import csv
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with open(pdf, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_embeddings_to_csv(vectorstore, csv_filename):
    """Save embeddings from a vectorstore to a CSV file."""
    num_vectors = vectorstore.index.ntotal  # Get the total number of stored vectors
    embeddings = [vectorstore.index.reconstruct(i) for i in range(num_vectors)]  # Reconstruct each vector

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = [str(i) for i in range(len(embeddings[0]))]  # Assuming all embeddings have the same length
        csv_writer.writerow(headers)

        for embedding in embeddings:
            csv_writer.writerow(embedding.tolist())

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    responses = []
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            responses.append("User: " + message.content)
        else:
            responses.append("Bot: " + message.content)
    return responses

def main():
    load_dotenv()
    openai_key = "insert your api key here"
    pdf_path = "/content/Ads cookbook .pdf"
    raw_text = get_pdf_text([pdf_path])
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks, openai_key)
    save_embeddings_to_csv(vectorstore, "embeddings.csv")
    conversation_chain = get_conversation_chain(vectorstore, openai_key)
    while True:
        user_question = input("Ask questions about your documents (type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        responses = handle_userinput(user_question, conversation_chain)
        for response in responses:
            print(response)

if __name__ == '__main__':
    main()