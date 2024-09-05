import os
import logging
from anthropic import Anthropic
from config import IDENTITY, TOOLS, MODEL, RAG_PROMPT
from dotenv import load_dotenv
from google_drive_utils import get_drive_service, get_documents, get_document_content
from embedding_utils import EmbeddingUtil, create_embeddings, create_faiss_index, search_similar
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.INFO)

class ChatBot:
    def __init__(self, session_state):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.anthropic = Anthropic(api_key=api_key)
        self.session_state = session_state
        self.drive_service = get_drive_service()
        self.documents = self.load_documents()  # Load all the documents
        self.embedding_util = EmbeddingUtil()
        self.embeddings = self.embedding_util.create_embeddings(self.documents)
        self.index = self.embedding_util.create_faiss_index(self.embeddings)

    def load_documents(self):
        files = get_documents(self.drive_service)
        documents = []
        for file in files:
            content = get_document_content(self.drive_service, file['id'])
            documents.append(content)
            logging.info(f"Loaded document: {file['name']}")
        logging.info(f"Total documents loaded: {len(documents)}")
        return documents

    def get_relevant_context(self, query):
        # Use embedding search to find relevant context for the user query
        similar_indices = self.embedding_util.search_similar(query, self.index, self.embeddings)
        context = "\n".join([self.documents[i] for i in similar_indices])

        # Extract the most relevant fee details from the document content
        if "fees" in query.lower() or "admission fees" in query.lower():
            fee_details = self.extract_fee_details(context)
            return fee_details if fee_details else "Sorry, I couldn't find the exact fee details in the documents."

        return context

    def extract_fee_details(self, document_content):
        # Simple example: look for fee-related words in the document content
        for line in document_content.splitlines():
            if "admission fee" in line.lower():
                return line  # Return the exact line containing fee details
        return None

    def generate_message(self, messages, max_tokens):
        try:
            logging.info(f"Sending request to Anthropic API:")
            logging.info(f"Model: {MODEL}")
            logging.info(f"System: {IDENTITY}")
            logging.info(f"Max tokens: {max_tokens}")
            logging.info(f"Messages: {messages}")
            logging.info(f"Tools: {TOOLS}")
            
            response = self.anthropic.messages.create(
                model=MODEL,
                system=IDENTITY,
                max_tokens=max_tokens,
                messages=messages,
                tools=TOOLS,
            )
            return response
        except Exception as e:
            logging.error(f"Anthropic API Error: {str(e)}")
            return {"error": str(e)}

    def process_user_input(self, user_input):
        # Always attempt to retrieve document context first
        context = self.get_relevant_context(user_input)
        
        if context:
            # If relevant context is found, construct a message using it
            self.session_state.messages.append({"role": "user", "content": user_input})
            response_message = self.generate_message(
                messages=self.session_state.messages,
                max_tokens=1024,  # Keep the response concise
            )
            
            if "error" in response_message:
                return f"An error occurred: {response_message['error']}"
            
            response_text = response_message.content[0].text if response_message.content else "I'm sorry, I couldn't find the information."
            self.session_state.messages.append({"role": "assistant", "content": response_text})
            return response_text
        else:
            # If no context found, provide a fallback general response
            return "I couldn’t find specific details related to your question in the documents. Could you provide more details, or check with the administration for precise information?"
