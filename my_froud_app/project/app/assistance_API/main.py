import os
from dotenv import load_dotenv
import openai
import requests
import json
from openai import OpenAI
import socket  # For checking internet connection
import time
import logging
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="AI-based Fraud Detector", page_icon=":detective:")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Custom CSS for design
st.markdown("""
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }
    .upload-section {
        text-align: center;
        margin-top: 40px;
        margin-bottom: 40px;
    }
    .upload-button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        border: none;
        border-radius: 50%;
        width: 150px;
        height: 150px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .upload-button:hover {
        background: linear-gradient(90deg, #92FE9D 0%, #00C9FF 100%);
        color: white;
        cursor: pointer;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #555555;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Model selection
model = "gpt-4o-mini"#gpt-4-1106-preview"

# == Hardcoded ids to be used once the first code run is done and the assistant was created
thread_id = "thread_K9iIgkd8J8w55bSOLo3n1h9U"
assis_id = "asst_TZrMZTdIN7ji0DiyBu2okNEI"

if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []





def check_internet_connection():
    """Checks if the device is connected to the internet."""
    try:
        # Attempt to resolve a public DNS name
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


# ==== Function definitions etc =====
def upload_to_openai(filepath):
    """Uploads a file to OpenAI API."""
    try:
        with open(filepath, "rb") as file:
            response = client.files.create(file=file.read(), purpose="assistants")
        return response.id
    except openai.error.APIConnectionError:
        st.error("Unable to connect to OpenAI API. Please check your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None



# === Sidebar: Multiple File Upload ===
with st.sidebar:
    st.markdown("<h2 style='color: #00C9FF;'>Upload Files for analysis ", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        " ",
        accept_multiple_files=True,
        help="Drag and drop files here or browse to upload.",
        key="file_upload"
    )
# Upload button
with st.sidebar:
    if st.button("transfer the Files to Open AI"):
        if not check_internet_connection():
            st.warning("No internet connection. Please check your connection and try again.")
        elif uploaded_files:
            for file_uploaded in uploaded_files:
                with open(file_uploaded.name, "wb") as f:
                    f.write(file_uploaded.getbuffer())
                file_id = upload_to_openai(file_uploaded.name)
                if file_id:
                    st.session_state.file_id_list.append(file_id)
                    st.write(f"Uploaded: {file_uploaded.name} (ID: {file_id})")
                else:
                    st.error(f"Failed to upload {file_uploaded.name}")

# # Display those file ids
# if st.session_state.file_id_list:
#     st.sidebar.write("Uploaded File IDs:")
#     for file_id in st.session_state.file_id_list:
#         st.sidebar.write(file_id)
#         # Associate each file id with the current assistant
#         assistant_file = client.beta.assistants.files.create(
#             assistant_id=assis_id, file_id=file_id
#         )

# Button to initiate the chat session
if st.sidebar.button("Start Chatting..."):
    if st.session_state.file_id_list:
        st.session_state.start_chat = True

        # Create a new thread for this chat session
        chat_thread = client.beta.threads.create()
        st.session_state.thread_id = chat_thread.id
        st.write("Thread ID:", chat_thread.id)
    else:
        st.sidebar.warning(
            "No files found. Please upload at least one file to get started."
        )

# Define the function to process messages with citations
def process_message_with_citations(message):
    """Extract content and annotations from the message and format citations as footnotes."""
    # Check if message has content and it's not empty
    if not message.content:
        return "No content available in the message"
    
    # Get the first content item (text or image)
    first_content = message.content[0]
    
    # Check if it's a text content type
    if first_content.type != "text":
        return "Message contains non-text content that cannot be processed"
    
    message_content = first_content.text
    annotations = message_content.annotations if hasattr(message_content, "annotations") else []
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(
            annotation.text, f" [{index + 1}]"
        )

        # Gather citations based on annotation attributes
        if file_citation := getattr(annotation, "file_citation", None):
            # Retrieve the cited file details (dummy response here since we can't call OpenAI)
            cited_file = {
                "filename": "card_transdata (original).csv"
            }  # This should be replaced with actual file retrieval
            citations.append(
                f'[{index + 1}] {file_citation.quote} from {cited_file["filename"]}'
            )
        elif file_path := getattr(annotation, "file_path", None):
            # Placeholder for file download citation
            cited_file = {
                "filename": "card_transdata (original).csv"
            }  # TODO: This should be replaced with actual file retrieval
            citations.append(
                f'[{index + 1}] Click [here](#) to download {cited_file["filename"]}'
            )  # The download link should be replaced with the actual download path

    # Add footnotes to the end of the message content
    full_response = message_content.value + "\n\n" + "\n".join(citations)
    return full_response

# the main interface ...
# App header
st.markdown("<h1 class='main-title'>Advanced Fraud Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>AI-powered insights to uncover hidden patterns and accelerate investigations.</p>", unsafe_allow_html=True)

# Check sessions
if st.session_state.start_chat:
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o-mini"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show existing messages if any...
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input for the user
    if prompt := st.chat_input("What's new?"):
        # Add user message to the state and display on the screen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id, role="user", content=prompt
        )

        # Create a run with additioal instructions
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assis_id,
            instructions="""Thoroughness and Attention to Detail: Carefully examine all available evidence, including financial records,
                            communication logs, and witness statements. Pay close attention to inconsistencies, anomalies, and suspicious patterns.Ability or expertise to utilize machine learning to it's full pottential.
                            Data Analysis Skills: Develop strong analytical skills to identify trends, correlations, and outliers in large datasets. Utilize data visualization tools to effectively present findings.
                            Communication and Reporting: Clearly and concisely document findings, investigative steps, and conclusions in reports for legal and regulatory purposes. Effectively communicate findings to stakeholders, including law enforcement, legal counsel, and management.
                            Ethical Conduct: Maintain the highest ethical standards throughout the investigation process. Respect the privacy and rights of all individuals involved.""",
        )

        # Show a spinner while the assistant is thinking...
        with st.spinner("Wait... Generating response..."):
            while run.status != "completed":
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id, run_id=run.id
                )
            # Retrieve messages added by the assistant
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )
            # Process and display assis messages
            assistant_messages_for_run = [
                message
                for message in messages
                if message.run_id == run.id and message.role == "assistant"
            ]

            for message in assistant_messages_for_run:
                full_response = process_message_with_citations(message=message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                with st.chat_message("assistant"):
                    st.markdown(full_response, unsafe_allow_html=True)

    else:
        # Promopt users to start chat
        st.write(
            "Please upload at least a file to get started by clicking on the 'Start Chat' button"
        )


# Footer
st.markdown("<div class='footer'>Powered by OpenAI and Streamlit</div>", unsafe_allow_html=True)