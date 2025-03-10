import os, json
import pandas as pd
from pipeline import pipe
from prompt import sentiment_agent_role, outcome_agent_role, assessor_agent_role
from openai import OpenAI

# Initialise client (assumes your API key is set in the environment)
API_KEY = os.getenv(input('Enter your environment variable: '))
client = OpenAI(api_key=API_KEY)
df = pd.read_csv('../output/transcripts_data.csv')

# Load response formats
with open("../config/response_format_assessor.json", "r") as f:
    assessor_response_format = json.load(f)
with open("../config/response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)
with open("../config/response_format_outcome.json", "r") as f:
    outcome_response_format = json.load(f)

# Load the optimized model parameters (ensuring they are converted to a dict)
model_params_data = pd.read_csv('../output/grid_search_results.csv')
best_model_params = model_params_data.loc[model_params_data['sse'].idxmin()].to_dict()
model_params_str = best_model_params['params']
model_params = json.loads(model_params_str.replace("'", '"'))  # Convert to dict if necessary

# List to collect pipeline outputs
pipeline_outputs = []
for idx, row in df.iterrows():
    member_body = row['member_body']
    print(f"Processing row {idx}...")
    # Call the pipeline function and capture its returned dictionary
    output = pipe(
        member_body=member_body,
        client=client,
        model_params=model_params,
        sentiment_agent_role=sentiment_agent_role,
        sentiment_response_format=sentiment_response_format,
        assessor_agent_role=assessor_agent_role,
        assessor_response_format=assessor_response_format,
        outcome_agent_role=outcome_agent_role,
        outcome_response_format=outcome_response_format
    )
    pipeline_outputs.append(output)

# Convert the list of dictionaries into a DataFrame
outputs_df = pd.DataFrame(pipeline_outputs)
overall_df = pd.concat([df.reset_index(drop=True), outputs_df], axis=1)
overall_df.to_csv('../output/overall_model_output.csv', index=False)
print("Overall model output saved to ../output/overall_model_output.csv")
