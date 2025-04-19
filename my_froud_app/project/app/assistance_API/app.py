import os
from dotenv import load_dotenv
import openai
import time
import logging


load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Model selection
model = "gpt-4o-mini"#gpt-4-1106-preview"

# Step 1: Upload File
# file_response = client.files.create(
#     filepath=open("/home/armstrong/my_froud_app/venv/bin/project/app/assistance_API/20250329_170951_processed_data.csv", "rb"),
#     purpose="assistants"

# )

with open("/home/armstrong/my_froud_app/venv/bin/project/app/assistance_API/sample number 2.csv", "rb") as file:
    file_response = client.files.create(
        file=file,
        purpose="assistants"  # or whatever purpose you need
    )


# Access file ID directly from the FileObject
file_id = file_response.id
if not file_id:
    raise ValueError("File upload failed. No file ID returned.")
print(f"Uploaded File ID: {file_id}")


    # == Hardcoded ids to be used once the first code run is done and the assistant was created
thread_id = "thread_K9iIgkd8J8w55bSOLo3n1h9U"
assis_id = "asst_TZrMZTdIN7ji0DiyBu2okNEI"




# Step 2: Create an Assistant with both File Search and Code Interpreter Tools
# try:
#     assistant = client.beta.assistants.create(
#         name="fraud detective",
#         instructions="""As a highly skilled fraud detective,
#         I possess a deep understanding of fraudulent activities and the ability to analyze complex data to uncover hidden patterns and anomalies.
#         My expertise encompasses advanced data analysis techniques, including statistical analysis, data mining, and machine learning,
#         which I utilize to identify suspicious activities and proactively prevent fraud.
#         I am proficient in investigative techniques such as fraud scheme recognition, evidence collection,
#         and interview techniques. Furthermore, I possess strong technical skills, including proficiency in fraud detection software and data visualization tools,
#         enabling me to effectively communicate findings and support legal and regulatory proceedings. My goal is to protect organizations and individuals from financial losses by mitigating the impact of fraud in today's increasingly digital world.""",
#         tools=[{"type": "code_interpreter"}, {"type": "file_search"}],  # Include both code_interpreter and file_search
#         tool_resources={
#             "code_interpreter": {
#                 "file_ids": [file_id]  # Only the code_interpreter needs file_ids
#             },
#             # No file_ids for file_search tool
#         },
#         model=model,
#     )
#     assistant_id = assistant.id
#     print(f"Assistant Created with ID: {assistant_id}")
# except Exception as e:
#     raise ValueError(f"Failed to create assistant: {e}")

























# Step 3: Create a Thread
# try:
#     thread = client.beta.threads.create()
#     thread_id = thread.id
#     print(f"Thread Created with ID: {thread_id}")
# except Exception as e:
#     raise ValueError(f"Failed to create thread: {e}")

# Step 4: Send a Message
message = "What are the trends within the data?"
try:
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    print(f"Message Sent: {message}")
except Exception as e:
    raise ValueError(f"Failed to send message: {e}")

# # Step 5: Run the Assistant
try:
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assis_id,
        instructions="Please include visualizations in your response if possible."
    )
    run_id = run.id
    print(f"Run Started with ID: {run_id}")
except Exception as e:
    raise ValueError(f"Failed to start run: {e}")


def wait_for_run_completion(client, thread_id, run_id, sleep_interval=5):
    """ Waits for a run to complete and prints the elapsed time. """
    while True:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.completed_at:
                elapsed_time = run.completed_at - run.created_at
                formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(f"Run completed in {formatted_elapsed_time}")
                logging.info(f"Run completed in {formatted_elapsed_time}")
                
                # Get messages here once Run is completed!
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                last_message = messages.data[0]
                
                # Check the content type before accessing it
                if hasattr(last_message.content[0], 'text'):
                    response = last_message.content[0].text.value
                    print(f"Assistant Response: {response}")
                elif hasattr(last_message.content[0], 'image'):
                    image_url = last_message.content[0].image.url
                    print(f"Assistant Response (Image): {image_url}")
                else:
                    print(f"Unknown response type: {last_message.content[0]}")
                break
        except Exception as e:
            logging.error(f"An error occurred while retrieving the run: {e}")
            break
        logging.info("Waiting for run to complete...")
        time.sleep(sleep_interval)


# # # == Run it
wait_for_run_completion(client=client, thread_id=thread_id, run_id=run.id)


# # === Check the Run Steps - LOGS ===
run_steps = client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run.id)
print(f"Run Steps --> {run_steps.data[0]}")























# {
#   "name": "analyze_fraud_transactions",
#   "description": "Identify potentially fraudulent transactions from uploaded transaction data.",
#   "strict": true,
#   "parameters": {
#     "type": "object",
#     "required": [
#       "uploaded_data",
#       "transaction_columns",
#       "sample_data"
#     ],
#     "properties": {
#       "uploaded_data": {
#         "type": "string",
#         "description": "Path to the uploaded CSV file containing transaction data."
#       },
#       "transaction_columns": {
#         "type": "array",
#         "description": "List of column names in the transaction data.",
#         "items": {
#           "type": "string",
#           "description": "Name of a transaction column."
#         }
#       },
#       "sample_data": {
#         "type": "array",
#         "description": "Sample rows from the transaction data for analysis.",
#         "items": {
#           "type": "object",
#           "description": "A sample transaction record.",
#           "properties": {
#             "transaction_id": {
#               "type": "string",
#               "description": "Unique identifier for the transaction"
#             },
#             "timestamp": {
#               "type": "string",
#               "description": "Date and time of the transaction"
#             },
#             "amount": {
#               "type": "number",
#               "description": "Amount of money involved in the transaction"
#             },
#             "sender_id": {
#               "type": "string",
#               "description": "Identifier for the sender's account"
#             },
#             "receiver_id": {
#               "type": "string",
#               "description": "Identifier for the receiver's account"
#             },
#             "transaction_type": {
#               "type": "string",
#               "description": "Type of transaction (e.g., credit, debit)"
#             },
#             "location": {
#               "type": "string",
#               "description": "Location or IP address where the transaction occurred"
#             },
#             "device_info": {
#               "type": "string",
#               "description": "Information about the device used for the transaction"
#             }
#           },
#           "required": [
#             "transaction_id",
#             "timestamp",
#             "amount",
#             "sender_id",
#             "receiver_id",
#             "transaction_type",
#             "location",
#             "device_info"
#           ],
#           "additionalProperties": false
#         }
#       }
#     },
#     "additionalProperties": false
#   }
# }