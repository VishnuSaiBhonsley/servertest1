from flask import Flask, request, jsonify,render_template
import google.generativeai as genai
from flask_cors import CORS
import os
import pdfplumber
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings import CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import cohere
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCnCmG44TfCZq27P9aVVX5ug1x_ovhb1kI"
if "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = "d1twBEVfMurf4bQf7O377CUbTmUvKgC5ugLg62F4"

# API Keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCnCmG44TfCZq27P9aVVX5ug1x_ovhb1kI")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "d1twBEVfMurf4bQf7O377CUbTmUvKgC5ugLg62F4")

# Initialize LLM models
gemini_llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")
cohere_llm = cohere.Client(COHERE_API_KEY)

# Initialize Embeddings (both Google and Cohere)
google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
cohere_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, user_agent="my-agent")

# Extract text from PDF function
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text()
    return all_text

# Load and split PDF text
pdf_text = extract_text_from_pdf('server/all.pdf')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_text(pdf_text)

# Initialize FAISS Vector Store with Cohere embeddings by default
vectorstore = FAISS.from_texts(documents, cohere_embeddings)
vectorstore.save_local("vectorstore.db")
retriever = vectorstore.as_retriever()

# Setup Chat History and Memory
memory = ConversationBufferMemory(memory_key="chat_history")
contextualize_q_system_prompt = """Introduce Yourself: Greet the user, summarize their query, and respond strictly based on the chat history.
Do not answer irrelevant questions. Provide service details related to the company's services concisely."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    gemini_llm, retriever, contextualize_q_prompt
)

# Define the template for the response
qa_template = """You are Weenee, a virtual assistant from Lollypop design company. Give concise answers to the point.
If you don't know the answer, say you don't know. Do not include special characters or emojis.
<context>{context}</context> Response:"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create Document and Retrieval Chains
doc_chain = create_stuff_documents_chain(gemini_llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

# Manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# RAG Chain with message history
conversation_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Function to decide which model to use based on user input
def prompt_model(text, session_id, model_choice="gemini"):
    if model_choice == "gemini":
        response = conversation_rag_chain.invoke({"input": text}, config={"configurable": {"session_id": session_id}})['answer']
    else:  # Use Cohere as a fallback
        response = cohere_llm.generate(prompt=text, max_tokens=100).generations[0].text
    return response

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('user_input')
    session_id = request.json.get('session_id')
    model_choice = request.json.get('model_choice')  # Choose the model: 'gemini' or 'cohere'
    print(model_choice)
    
    try:
        # Get relevant documents using the retriever
        retrieved_documents = retriever.get_relevant_documents(user_input)
        embedded_context = [doc.page_content for doc in retrieved_documents]
        
        # Print embedded context for debugging (optional)
        # for context in embedded_context:
        #     print(context)

        # Generate response using the chosen model
        response = prompt_model(user_input, session_id, model_choice)
        return jsonify(response=response)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500
@app.route('/')
def main():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='10.20.100.18',port=5000)
