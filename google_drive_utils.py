import os
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.helpers import credentials_from_session

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=SCOPES
            )
            
            flow.run_local_server(port=8501)
            
            creds = flow.credentials

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def get_documents(service):
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        spaces='drive',
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def get_document_content(service, file_id):
    document = service.files().export(fileId=file_id, mimeType='text/plain').execute()
    return document.decode('utf-8')