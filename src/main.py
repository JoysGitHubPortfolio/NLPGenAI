from pipeline import *
from prompt import sentiment_agent_role, outcome_agent_role, assessor_agent_role

with open("../config/response_format_assessor.json", "r") as f:
    assessor_response_format = json.load(f)
with open("../config/response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)
with open("../config/response_format_outcome.json", "r") as f:
    outcome_response_format = json.load(f)

df = pd.read_csv('../output/transcripts_data.csv')
text = df['member_body'][0]
print(text)

# Call the model passing all relevant schemas and input
pipe(member_body=text,
     client=client,
     model_params=model_params,

     sentiment_agent_role=sentiment_agent_role,
     sentiment_response_format=sentiment_response_format,
     
     assessor_agent_role=assessor_agent_role,
     assessor_response_format=assessor_response_format,
     
     outcome_agent_role=outcome_agent_role,
     outcome_response_format=outcome_response_format)