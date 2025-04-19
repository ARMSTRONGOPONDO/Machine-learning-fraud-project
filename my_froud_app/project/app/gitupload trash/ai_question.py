import os
from openai import OpenAI
from flask import  flash, redirect, url_for

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import pandas as pd

# Set up OpenAI API key

# Path to the folder where processed data is stored
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

def load_latest_data():
    """Load the most recently processed data file."""
    try:
        if not os.path.exists(PROCESSED_FOLDER):
            raise FileNotFoundError("Processed folder does not exist.")

        files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith("_processed_data.csv")]
        if not files:
            raise FileNotFoundError("No processed data files found.")

        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(PROCESSED_FOLDER, f)))
        return pd.read_csv(os.path.join(PROCESSED_FOLDER, latest_file))
    except Exception as e:
        print(f"Error: {e}")
        return None

def ask_ai_about_data(question):
    """
    Use OpenAI API to analyze the data and answer a question about it.
    Args:
        question (str): The user's question.
    Returns:
        str: The AI's answer.
    """
    # Load the latest data
    df = load_latest_data()
    if df is None:
        return "No data available to analyze."

    # Prepare a summary of the dataset
    data_summary = df.describe(include='all').to_string()

    # Construct the context for the AI
    context = f"""
    You are a data analysis assistant. Analyze the following dataset summary and answer the user's question:

    Dataset Summary:
    {data_summary}

    Question: {question}

    Please provide a detailed, structured, and insightful response. Start with a general overview of the dataset, 
    then discuss any significant trends, correlations, or anomalies you find. Conclude with a direct answer to the user's question.
    """

    # Query the OpenAI API (using the new ChatCompletion method)
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": context},
        ],
        max_tokens=500,  # Increased token limit for detailed responses
        temperature=0.7,  # Increased temperature for more detailed and creative responses
        top_p=0.9  # Nucleus sampling for diversity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    





    # interface discription 
                # AI Q&A Section
        # html.Div([
        #     html.H4("Ask AI About Your Data", className="mb-3 text-white"),
        #     dcc.Input(
        #         id="user-question",
        #         type="text",
        #         placeholder="Type your question here...",
        #         className="mb-3",
        #         style={"width": "100%"}
        #     ),
        #     dbc.Button("Ask", id="ask-button", color="primary", className="mb-3"),
        #     html.Div(id="ai-answer", className="mt-3 text-white")
        # ], className="mb-4"),




    #from ai_question import ask_ai_about_data  # Make sure this import is correct
    #         # Callback for AI Q&A
    # @dash_app.callback(
    #     Output("ai-answer", "children"),
    #     [Input("ask-button", "n_clicks")],
    #     [State("user-question", "value")]
    # )
    # def handle_ai_question(n_clicks, question):
    #     if not question or n_clicks is None:
    #         return "Please enter a question."
    #     return ask_ai_about_data(question) 