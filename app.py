from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
import google.generativeai as genai
from flask_cors import CORS
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, ConversationalRetrievalChain, LLMChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import ChatCohere
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import BaseChatMessageHistory
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.retrievers import CohereRagRetriever
import cohere
from dotenv import load_dotenv
from search import *

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def ingest_data(pdf_path):
        # Extract and split text
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_text(all_text)

        # Initialize Google embeddings
        google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize FAISS Vector Store with Google embeddings
        Data = FAISS.from_texts(
            texts=documents, 
            embedding=google_embeddings
        )
        Data.save_local("vector_db")


class GoogleWorkflow:
    """
    Class to handle Google-related operations including loading data,
    embedding, and retrieval using Google Generative AI.
    """

    def __init__(self, api_key, pdf_path):
        """
        Initialize GoogleWorkflow with API key and PDF path.

        Parameters:
        api_key (str): Google API key.
        pdf_path (str): Path to the PDF document for loading data.
        """
        self.api_key = api_key
        self.pdf_path = pdf_path
        self.llm = ChatGoogleGenerativeAI(api_key=self.api_key, model="gemini-1.5-flash")
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.history_aware_retriever = self.create_history_aware_retriever()
        self.conversation_rag_chain = self.create_conversation_chain()



    def create_history_aware_retriever(self):
        """
        Create a history-aware retriever using the LLM and the vector store.

        Returns:
        History-aware retriever instance.
        """
        contextualize_q_system_prompt = """   Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])



        # check if the vector store exists
        if os.path.exists("vector_DB"):

            vectore_store = FAISS.load_local(
                folder_path= "vector_db",
                embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization= True
            )

        else:
            ingest_data('test.pdf')

            vectore_store = FAISS.load_local(
                folder_path="vector_db",
                embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization= True
            )


        # Create history-aware retriever
        retriever = vectore_store.as_retriever()
        return create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

    def create_conversation_chain(self):
        """
        Create a conversation RAG chain.

        Returns:
        RunnableWithMessageHistory instance for conversation handling.
        """
        qa_template = """You are Weenee, a virtual assistant from terralogic company. Introduce Yourself: Greet the user, summarize their query, and respond strictly based on the chat history.
        Do not answer irrelevant questions. Provide service details related to the company's services concisely. the size of answer less 60 words  Give concise answers to the point.
        If you don't know the answer, say you don't know. Do not include special characters or emojis.Provide details related to the company's services concisely. the size of answer less 60 words
        <context>{context}</context> Response:"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create Document and Retrieval Chains
        doc_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(self.history_aware_retriever, doc_chain)

        # Manage chat history
        self.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def prompt_model(self, text, session_id):
        """
        Generate a response using the Google model based on user input.

        Parameters:
        text (str): User's query input.
        session_id (str): Session ID for maintaining conversation history.
        model_choice (str): The model choice (default is "gemini").

        Returns:
        str: Response from the Google model.
        """

        response = self.conversation_rag_chain.invoke({"input": text}, config={"configurable": {"session_id": session_id}})['answer']
        
        return response
    
class CohereWorkflow:
    """
    Class to handle Cohere-related operations and serve as a chatbot assistant.
        """
    def __init__(self, api_key, pdf_path='terralogic.pdf'):
        """
        Initialize CohereWorkflow with API key and PDF path.

        Parameters:
        api_key (str): Cohere API key.
        pdf_path (str): Path to the PDF document for loading data.
        """
        self.api_key = api_key
        self.pdf_path = pdf_path
        self.llm = ChatCohere(model='command-r')  # Initialize Cohere model
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.history_aware_retriever = self.create_history_aware_retriever()
        self.conversation_rag_chain = self.create_conversation_chain()

    def create_history_aware_retriever(self):
        """
        Create a history-aware retriever using the Cohere model and the vector store.

        Returns:
        History-aware retriever instance.
        """
        contextualize_q_system_prompt = """   Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        if os.path.exists("vector_DB"):

            vectore_store = FAISS.load_local(
                folder_path= "vector_db",
                embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization= True
            )

        else:
            ingest_data('test.pdf')

            vectore_store = FAISS.load_local(
                folder_path="vector_db",
                embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                allow_dangerous_deserialization= True
            )


        # Create history-aware retriever
        retriever = vectore_store.as_retriever()
        return create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

    def create_conversation_chain(self):
        """
        Create a conversation RAG chain.

        Returns:
        RunnableWithMessageHistory instance for conversation handling.
        """
        qa_template = """
        ##Task and Context
        You are Jimmy, the virtual assistant for Terralogic. Your goal is to provide helpful, concise responses related to Terralogic's services and the current conversation context. Greet the user, briefly summarize their question, and respond directly based on the information available in the chat history.
        
        ##Response Format
        Limit responses to 60 words or fewer.
        If a question is irrelevant or outside your knowledge, say "I don't know."
        Focus only on Terralogic-related topics; do not answer off-topic questions.
        Do not use special characters or emojis.
        
        ##Context
        <context>{context}</context>

        Response:"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create Document and Retrieval Chains
        doc_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(self.history_aware_retriever, doc_chain)

        # Manage chat history
        self.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def prompt_model(self, user_input, session_id):
        """
        Generate a response using the Cohere model based on user input.

        Parameters:
        user_input (str): User's query input.
        session_id (str): Session ID for maintaining conversation history.
        model_choice (str): The model choice (default is "cohere").

        Returns:
        str: Response from the Cohere model.
        """


        response = self.conversation_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})['answer']
        
        return response







# Initialize workflows with PDF paths
google_workflow = GoogleWorkflow(GOOGLE_API_KEY, 'terralogic.pdf')
cohere_workflow = CohereWorkflow(COHERE_API_KEY, 'terralogic.pdf')

# Function to decide which model to use based on user input
def decide_model(user_input, session_id, model_choice="google"):
    """
    Decide which model to use based on user input.
    
    Parameters:
    user_input (str): User's query input.
    session_id (str): Session ID for maintaining conversation history.
    model_choice (str): The model to use ('google' or 'cohere').
    
    Returns:
    str: Response from the selected model.
    """
    if model_choice == "google":
        print(model_choice)
        return google_workflow.prompt_model(user_input, session_id)
    else:  # Use Cohere as a fallback
        print(model_choice)
        return cohere_workflow.prompt_model(user_input, session_id)
def format_response(qa_data):
        if not qa_data:
            return "No data available."

        # Get the first question and answer
        first_qa = qa_data[0]
        first_question = first_qa['question']
        first_answer = first_qa['answer']
        
        # Get remaining questions for options
        options = [item['question'] for item in qa_data[1:]]  # All questions except the first one

        # Format the response
        response = {
            first_question:{"response": first_answer,'options': options}
        }
        
        return response

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint to handle user queries.
    
    Returns:
    JSON response containing the model's output.
    """
    
    user_input = request.json.get('user_input')
    session_id = request.json.get('session_id')
    model_choice = request.json.get('model_choice', 'google')  # Default to Google

    try:
        # Generate response using the chosen model
        response = decide_model(user_input, session_id, model_choice)
        return jsonify(response=response)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500
    
@app.route('/getresponses', methods=['POST'])
def get_responses():
    try:
        user_input = request.json.get('user_input')
        top_faqs, top_scores = faq_search(user_input)

        # Format and display results
        def format_response(qa_data):
            if not qa_data:
                return "No data available."

            first_qa = qa_data[0]
            first_question = str(first_qa['question']).lower()
            first_answer = first_qa['answer']
            options = [item['question'] for item in qa_data[1:]]
            
            return {
                first_question: {"response": first_answer, 'options': options}
            }
        
        response = format_response(top_faqs)
        return jsonify(response)  # Return the response directly
        
    except Exception as e:
        print(f"Error: {e}")  # Log error for debugging
        return jsonify(error=str(e)), 500
@app.route('/')
def main():
    """
    Main endpoint serving the HTML interface.
    
    Returns:
    Rendered HTML template.
    """
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,host="192.168.55.89")

##########################################################################################################################

