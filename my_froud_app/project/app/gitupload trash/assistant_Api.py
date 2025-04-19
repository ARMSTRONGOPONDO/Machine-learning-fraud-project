from openai import OpenAI
client = OpenAI()

assistant = client.beta.assistants.create(
name="Math Tutor",
instructions="You are a data analytics expert. Write and run code to answer quations about data .",
tools=[{"type": "code_interpreter,file_search"}],
model="gpt-4o",
)
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="""
    I have uploaded a dataset called `card_transdata.csv`, which contains the following columns:
    - used_chip
    - distance_from_home
    - online_order
    - distance_from_last_transaction
    - repeat_retailer
    - ratio_to_median_purchase_price
    - used_pin_number
    - Fraud Label (1 for fraudulent, 0 for non-fraudulent)

    Please:
    - Analyze the data and identify patterns or anomalies indicative of fraud.
    - Suggest metrics or features that could improve the detection process.
    - Provide a detailed analysis based on the data.
    """
)


