from openai import OpenAI
from agent import MakeMyAgent, json
from prompt import sentiment_agent_role, assessor_agent_role, user_input
import os


# check for consistence of output between different agents
def is_improvement_required(json_assessor_output, json_sentiment_output):
    try:
        improvement = json_assessor_output['improvement']
        if (improvement['negative'] != json_sentiment_output['negative'] or 
            improvement['neutral'] != json_sentiment_output['neutral'] or
            improvement['positive'] != json_sentiment_output['positive']):
            return True
        else:
            return False
    except:
        print('Could not obtain dictionary')
        return None
    
# Initialize OpenAI client once
API_KEY = os.getenv(input('Enter your environment variable: '))
client = OpenAI(api_key=API_KEY) 

with open("response_format_assessor.json", "r") as f:
    assessor_response_format = json.load(f)
with open("response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)

# Define model parameters
model_params = {
    "temperature": 1,
    "max_tokens": 2048,
    "top_p": 1,
    "frequency_penalty": 1,
    "presence_penalty": 1
}

# Get my sentiment analysis
sentiment_agent = MakeMyAgent(
    model = "gpt-4o-mini",
    client = client, 
    model_params = model_params,
    role = sentiment_agent_role,
    response_format = sentiment_response_format,
    user_input = user_input
)
sentiment_output = sentiment_agent.run()
json_sentiment_output = json.loads(sentiment_output)

# Engineers can debug with quality analysis
assessor_agent = MakeMyAgent(
    model = "gpt-4o-mini",
    client = client, 
    model_params = model_params,
    role = assessor_agent_role,
    response_format = assessor_response_format,
    user_input = sentiment_output
)
assessor_output = assessor_agent.run()
json_assessor_output = json.loads(assessor_output)

# check instance of improvement
improvement = is_improvement_required(json_assessor_output, json_sentiment_output)
print(json_sentiment_output)
print(json_assessor_output['improvement'])
print(improvement)