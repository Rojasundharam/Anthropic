# Eva: Acme Insurance Company's AI Assistant

This project implements an AI-powered chatbot named Eva for Acme Insurance Company. Eva can provide information about insurance offerings and help customers get quotes.

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google Drive API credentials and place the `credentials.json` file in the project root
4. Create a `.env` file with your Anthropic API key and the path to your Google credentials
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Files

- `app.py`: The main Streamlit application
- `chatbot.py`: Implements the ChatBot class
- `config.py`: Contains configuration settings and prompts
- `google_drive_utils.py`: Utilities for interacting with Google Drive
- `embedding_utils.py`: Utilities for creating and searching document embeddings

## Usage

Once the app is running, you can interact with Eva by typing messages in the chat interface. Eva will respond based on the information in the connected Google Drive documents and her training.