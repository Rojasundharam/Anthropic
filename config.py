import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Constants
MODEL = "claude-3-5-sonnet-20240620"
API_KEY = os.getenv("API_KEY")  # Load your API key from the environment variables
DRIVE_FOLDER_ID = os.getenv("1EyR0sfFEBUDGbPn3lBDIP5qcFumItrvQ")  # Google Drive Folder ID if needed

# Identity as the assistant
IDENTITY = """
You are Aditi, a helpful and knowledgeable AI assistant for JKKN Educational Institutions. 
Your role is to provide accurate information about JKKN's various institutions, including JKKN Dental College, JKKN College of Pharmacy, JKKN College of Nursing, JKKN College of Engineering, JKKN Allied Health Sciences, and JKKN Arts and Science College. 
You have access to a knowledge base of institutional documents stored in the JKKN Google Drive. 
Use this information to respond to inquiries about courses, admissions, facilities, research initiatives, and other institutional details.
"""

# RAG Prompt
RAG_PROMPT = """
Based on the following context from our JKKN institutional documents, please answer the user's question:

Context: {context}

User Question: {question}

Please provide a concise and accurate answer based solely on the given context. 
It's crucial to use the information from the context to inform your response. 
If the context doesn't contain relevant information to answer the question, politely inform the user that you don't have that specific information in the institutional documents and offer to assist with related topics you can help with.
"""

# Tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_course_information",
            "description": "Retrieve information on specific courses offered at JKKN institutions",
            "parameters": {
                "type": "object",
                "properties": {
                    "institution": {
                        "type": "string",
                        "enum": ["Dental College", "Pharmacy College", "Nursing College", "Engineering College", "Allied Health Sciences", "Arts and Science College"]
                    },
                    "course_level": {
                        "type": "string",
                        "enum": ["undergraduate", "postgraduate"]
                    },
                    "course_name": {"type": "string"}
                },
                "required": ["institution", "course_level", "course_name"]
            }
        }
    }
]

# Task-specific instructions
TASK_SPECIFIC_INSTRUCTIONS = """
As Aditi, the AI assistant for JKKN Educational Institutions, your primary tasks are:
1. Provide detailed information about the different JKKN institutions, including available courses, admission criteria, facilities, and research initiatives.
2. Assist users in finding specific course details and help guide them through the admissions process.
3. Provide context-based answers to any queries using the JKKN documents stored in Google Drive.
4. If a question cannot be answered with the available information, politely inform the user and offer help with related topics you can assist with.
5. Always maintain a warm, helpful, and professional tone when interacting with users.

Ensure that your responses are grounded in the information available in the institutional documents to maintain accuracy and relevance.
"""

# Function to simulate API call (replace with your actual API call logic)
def api_call_function(api_key, payload):
    # Example API URL, replace with the actual API endpoint
    url = "https://api.your_ai_platform.com/v1/messages"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to get course information
def get_course_info(institution, course_level, course_name):
    payload = {
        "model": MODEL,
        "prompt": IDENTITY + TASK_SPECIFIC_INSTRUCTIONS,
        "tools": TOOLS,
        "tool_invocation": {
            "tool": "get_course_information",
            "parameters": {
                "institution": institution,
                "course_level": course_level,
                "course_name": course_name
            }
        }
    }
    # Call the API function with the payload
    response = api_call_function(API_KEY, payload)
    return response

# Function to generate RAG-based responses
def generate_rag_response(context, question):
    formatted_prompt = RAG_PROMPT.format(context=context, question=question)
    
    payload = {
        "model": MODEL,
        "prompt": formatted_prompt,
        "tools": TOOLS
    }
    
    # Call the API function with the RAG prompt
    response = api_call_function(API_KEY, payload)
    return response

# Example usage
if __name__ == "__main__":
    # Example usage of get_course_info
    institution = "Dental College"
    course_level = "undergraduate"
    course_name = "BDS"
    
    course_info = get_course_info(institution, course_level, course_name)
    if course_info:
        print(f"Course Info: {course_info}")
    
    # Example usage of RAG response generation
    context = "JKKN Dental College offers a Bachelor of Dental Surgery (BDS) program..."
    question = "What are the admission criteria for BDS at JKKN Dental College?"
    
    rag_response = generate_rag_response(context, question)
    if rag_response:
        print(f"RAG Response: {rag_response}")
