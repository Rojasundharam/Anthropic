from anthropic import Anthropic
from config import IDENTITY, TOOLS, MODEL, RAG_PROMPT
from dotenv import load_dotenv
from google_drive_utils import get_drive_service, get_documents, get_document_content
from embedding_utils import create_embeddings, create_faiss_index, search_similar
import numpy as np

load_dotenv()

class ChatBot:
    def __init__(self, session_state):
        self.anthropic = Anthropic()
        self.session_state = session_state
        self.drive_service = get_drive_service()
        self.documents = self.load_documents()
        self.embeddings = create_embeddings(self.documents)
        self.index = create_faiss_index(self.embeddings)

    def load_documents(self):
        files = get_documents(self.drive_service)
        documents = []
        for file in files:
            content = get_document_content(self.drive_service, file['id'])
            documents.append(content)
        return documents

    def get_relevant_context(self, query):
        similar_indices = search_similar(query, self.index, self.embeddings)
        context = "\n".join([self.documents[i] for i in similar_indices])
        return context

    def generate_message(self, messages, max_tokens):
        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                system=IDENTITY,
                max_tokens=max_tokens,
                messages=messages,
                tools=TOOLS,
            )
            return response
        except Exception as e:
            return {"error": str(e)}

    def process_user_input(self, user_input):
        context = self.get_relevant_context(user_input)
        rag_message = RAG_PROMPT.format(context=context, question=user_input)
        
        self.session_state.messages.append({"role": "user", "content": rag_message})
        
        response_message = self.generate_message(
            messages=self.session_state.messages,
            max_tokens=2048,
        )
        
        if "error" in response_message:
            return f"An error occurred: {response_message['error']}"
        
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
            # Implement the logic to retrieve course information
            # This is a placeholder implementation
            return f"Here's information about {func_params['course_name']} at {func_params['institution']} (Level: {func_params['course_level']}): [Course details would be retrieved here]"
        else:
            return f"Tool use requested: {func_name} with parameters {func_params}"