from openai import OpenAI
from agent import MakeMyAgent
from smart_functions import SmartFunction
from prompt import sentiment_agent_role, outcome_agent_role, assessor_agent_role
import pandas as pd
import numpy as np
import os, json 

# Initialise client by passing key known to you & Initialise SmartFunction & parameters
API_KEY = os.getenv(input('Enter your environment variable: '))
client = OpenAI(api_key=API_KEY) 
sf = SmartFunction()

with open("../config/response_format_assessor.json", "r") as f:
    assessor_response_format = json.load(f)
with open("../config/response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)
with open("../config/response_format_outcome.json", "r") as f:
    outcome_response_format = json.load(f)

# Define model parameters
model_params = {
    "temperature": 0.5,
    "max_tokens": 2048,
    "top_p": 0.8,
    "frequency_penalty": 1,
    "presence_penalty": 1
}

# Load data
df = pd.read_csv('../output/transcripts_labelled.csv')
df = df.dropna(subset=['ground_positive', 'ground_neutral', 'ground_negative'])
print(df.head())

model_sentiments = {}
limit = len(df)
for transcript_id, member_body in zip(df['transcript_id'][0:limit], 
                                      df['member_body'][0:limit]):
    print(member_body)

    sentiment_agent = MakeMyAgent(
        model = "gpt-4o-mini",
        client = client, 
        model_params = model_params,
        role = sentiment_agent_role,
        response_format = sentiment_response_format,
        user_input = member_body
    )
    predicted_sentiment = sentiment_agent.run()
    json_predicted_sentiment = json.loads(predicted_sentiment)

    predictions = {}
    predictions['positive'] = json_predicted_sentiment['positive']
    predictions['neutral'] = json_predicted_sentiment['neutral']
    predictions['negative'] = json_predicted_sentiment['negative']
    model_sentiments[transcript_id] = predictions


# Convert dictionary to DataFrame
predictions_df = pd.DataFrame.from_dict(model_sentiments, orient='index')
predictions_df.reset_index(inplace=True)
predictions_df.rename(columns={'index': 'transcript_id', 
                               'positive': 'predicted_positive', 
                               'neutral': 'predicted_neutral', 
                               'negative': 'predicted_negative'}, inplace=True)

# Get SSE as metric for model
df = df.merge(predictions_df, on='transcript_id', how='left')
df['sse'] = df.apply(lambda row: (row['ground_positive'] - row['predicted_positive'])**2 + 
                                (row['ground_neutral'] - row['predicted_neutral'])**2 + 
                                (row['ground_negative'] - row['predicted_negative'])**2, axis=1)
df['psuedo_accuracy'] = (1 - df['sse'])/3
df.to_csv('../output/transcripts_psuedo_optimisation.csv')

model_sse = np.sum(df['sse'])
model_pseudo_accuracy = np.mean(df['psuedo_accuracy'])

print('SSE', model_sse)
print('Pseudo Accuracy', model_pseudo_accuracy)