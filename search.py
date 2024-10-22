import os
import json
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths for the cached files
PDF_PATH = 'FAQ_data/faqs.pdf'
EMBEDDINGS_PATH = 'FAQ_data/faq_embeddings.npz'
FAQ_JSON_PATH = 'FAQ_data/faqs_from_pdf.json'

# Global variables to hold precomputed FAQs and embeddings
faqs = None
faq_embeddings = None

def embed_sentences(sentences):
    """
    Embed a list of sentences using the SentenceTransformer model.
    """
    return model.encode(sentences)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return pages

def split_pdf_text_into_faqs_multiline(pdf_text):
    """
    Split the extracted PDF text into a list of FAQ entries, handling multi-line questions and answers.
    """
    faqs = []
    current_faq = {"question": "", "answer": ""}
    question_lines = []
    collecting_question = False

    for page in pdf_text:
        lines = page.split('\n')
        for line in lines:
            if 'https' in line or 'FAQ' in line:
                continue
            line = line.strip()
            if line.lower().startswith(("what", "how", "why", "when", "which", "where", "do", "does", "is")):
                if current_faq["question"] and current_faq["answer"]:
                    faqs.append(current_faq)
                    current_faq = {"question": "", "answer": ""}
                if not line.endswith('?'):
                    question_lines.append(line)
                    collecting_question = True
                else:
                    current_faq = {"question": line, "answer": ""}
                    collecting_question = False
            elif collecting_question:
                question_lines.append(line)
                if line.endswith('?'):
                    question = " ".join(question_lines)
                    current_faq = {"question": question, "answer": ""}
                    question_lines = []
                    collecting_question = False
            else:
                if not current_faq["question"]:
                    continue
                current_faq["answer"] += line.strip() + " "

    if current_faq["question"] and current_faq["answer"]:
        faqs.append(current_faq)

    return faqs

def load_faq_data():
    """
    Load FAQs and embeddings from precomputed files if available.
    Otherwise, extract from PDF and compute embeddings.
    """
    global faqs, faq_embeddings

    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAQ_JSON_PATH):
        # Load precomputed FAQs and embeddings
        with open(FAQ_JSON_PATH, 'r') as f:
            faqs = json.load(f)
        data = np.load(EMBEDDINGS_PATH)
        faq_embeddings = data['faq_embeddings']
        print("Loaded precomputed FAQs and embeddings.")
    else:
        # Extract text from PDF and compute embeddings
        print("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(PDF_PATH)
        faqs = split_pdf_text_into_faqs_multiline(pdf_text)
        
        with open(FAQ_JSON_PATH, 'w') as f:
            json.dump(faqs, f, indent=4)
        
        print("Computing embeddings...")
        faq_questions = [faq["question"] for faq in faqs]
        faq_embeddings = embed_sentences(faq_questions)
        
        np.savez(EMBEDDINGS_PATH, faq_embeddings=faq_embeddings)
        print("FAQs and embeddings saved.")

def faq_search(user_question, top_n=3, mode='cosine'):
    """
    Perform semantic search between the user question and the FAQs.
    """
    user_embedding = embed_sentences([user_question])

    if mode == 'cosine':
        similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        top_faqs = [faqs[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

    return top_faqs, top_scores

# Load FAQ data on startup
load_faq_data()

# Example usage
if __name__ == "__main__":
    user_query = "IT infrastructure"
    top_faqs, top_scores = faq_search(user_query)

    print("**********************************")
    print("User Query: ", user_query)
    print("**********************************")
    for idx, faq in enumerate(top_faqs):
        print(f"Top {idx + 1}:\nQ: {faq['question']}\nA: {faq['answer']}\nScore: {top_scores[idx]}\n")
