from openai import OpenAI
from agent import MakeMyAgent
from smart_functions import SmartFunction
from prompt import sentiment_agent_role, outcome_agent_role, assessor_agent_role
import os, json, pandas as pd, numpy as np



# Initialise client by passing key known to you & Initialise SmartFunction
# API_KEY = os.getenv(input('Enter your environment variable: '))
# client = OpenAI(api_key=API_KEY) 
# sf = SmartFunction()

with open("../config/response_format_assessor.json", "r") as f:
    assessor_response_format = json.load(f)
with open("../config/response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)
with open("../config/response_format_outcome.json", "r") as f:
    outcome_response_format = json.load(f)

# Choose the parameters where the SSE is lowest & set globally
model_params_data = pd.read_csv('../output/grid_search_results.csv')
best_model_params = model_params_data.loc[model_params_data['sse'].idxmin()]
model_params = best_model_params
print(model_params)

# # Get my sentiment analysis
# sentiment_agent = MakeMyAgent(
#     model = "gpt-4o-mini",
#     client = client, 
#     model_params = model_params,
#     role = sentiment_agent_role,
#     response_format = sentiment_response_format,
#     user_input = member_body
# )
# sentiment_output = sentiment_agent.run()
# json_sentiment_output = json.loads(sentiment_output)
# print(json_sentiment_output)

# # Engineers can debug with quality analysis
# assessor_agent = MakeMyAgent(
#     model = "gpt-4o-mini",
#     client = client, 
#     model_params = model_params,
#     role = assessor_agent_role,
#     response_format = assessor_response_format,
#     user_input = sentiment_output
# )
# assessor_output = assessor_agent.run()
# json_assessor_output = json.loads(assessor_output)
# print(json_assessor_output)

# # check instance of improvement
# sentiment_improvement = sf.is_improvement_required(json_sentiment_output, json_assessor_output)
# print(sentiment_improvement)
# print()


# # Create another agent to check if customer outcome as reached
# outcome_agent = MakeMyAgent(
#     model = "gpt-4o-mini",
#     client = client, 
#     model_params = model_params,
#     role = outcome_agent_role,
#     response_format = outcome_response_format,
#     user_input = member_body
# )
# outcome_output = outcome_agent.run()
# json_outcome_output = json.loads(outcome_output)
# print(json_outcome_output)

# outcome_assessor_agent = MakeMyAgent(
#     model = "gpt-4o-mini",
#     client = client,
#     model_params = model_params,
#     role = assessor_agent_role,
#     response_format = assessor_response_format,
#     user_input = outcome_output
# )
# outcome_assesor_output = outcome_assessor_agent.run()
# json_outcome_assessor_output = json.loads(outcome_assesor_output)
# print(json_outcome_assessor_output)

# # check instance of improvement
# outcome_improvement = sf.is_improvement_required(json_outcome_output, json_outcome_assessor_output)
# print(outcome_improvement)
# print()