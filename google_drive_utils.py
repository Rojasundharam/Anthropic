import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import streamlit as st

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = None
    if 'google_auth_token' in st.session_state:
        creds = Credentials.from_authorized_user_info(st.session_state['google_auth_token'], SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            st.write("Please visit this URL to authorize the application:")
            st.markdown(f"[Authorize]({auth_url})")
            auth_code = st.text_input("Enter the authorization code:")
            
            if auth_code:
                try:
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    st.session_state['google_auth_token'] = creds.to_json()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    return None

    if creds:
        return build('drive', 'v3', credentials=creds)
    else:
        return None

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