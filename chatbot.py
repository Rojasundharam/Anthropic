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
        self.documents = self.load_documents()
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
        similar_indices = self.embedding_util.search_similar(query, self.index, self.embeddings)
        context = "\n".join([self.documents[i] for i in similar_indices])
        
        # Limit context to first 1000 characters (adjust this based on requirements)
        max_context_length = 1000
        truncated_context = context[:max_context_length]

        return truncated_context

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
        # Get truncated context
        context = self.get_relevant_context(user_input)
        
        # Create a concise prompt using the truncated context
        rag_message = RAG_PROMPT.format(context=context, question=user_input)
        
        # Add the user message to session state
        self.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate a response with a limited token count to keep it concise
        response_message = self.generate_message(
            messages=self.session_state.messages,
            max_tokens=1024,  # Reduced token count for concise responses
        )
        
        # Check if there's an error
        if "error" in response_message:
            return f"An error occurred: {response_message['error']}"
        
        # Handle the response type
        if response_message.content[0].type == "text":
            response_text = response_message.content[0].text
            self.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
            return response_text
        elif response_message.content[-1].type == "tool_use":
            return self.handle_tool_use(response_message.content[-1].function_name, response_message.content[-1].parameters)
        else:
            raise Exception("An error occurred: Unexpected response type")

    def handle_tool_use(self, func_name, func_params):
        if func_name == "get_course_information":
            institution = func_params['institution']
            course_level = func_params['course_level']
            course_name = func_params['course_name']
            
            # Example course information database
            course_info = {
                "Dental College": {
                    "undergraduate": {
                        "Bachelor of Dental Surgery": "5-year program focusing on oral health and dental procedures."
                    },
                    "postgraduate": {
                        "Master of Dental Surgery": "3-year specialized program in various dental disciplines."
                    }
                },
                "Engineering College": {
                    "undergraduate": {
                        "B.Tech in Computer Science": "4-year program covering software development, algorithms, and computer systems."
                    },
                    "postgraduate": {
                        "M.Tech in Structural Engineering": "2-year advanced program in structural design and analysis."
                    }
                }
            }
            
            if institution in course_info and course_level in course_info[institution] and course_name in course_info[institution][course_level]:
                return f"Information about {course_name} at JKKN {institution} ({course_level} level): {course_info[institution][course_level][course_name]}"
            else:
                return f"I'm sorry, I don't have specific information about the {course_name} course at JKKN {institution} for the {course_level} level. Please check the JKKN website or contact the admissions office for the most up-to-date information."
        else:
            return f"Tool use requested: {func_name} with parameters {func_params}"
