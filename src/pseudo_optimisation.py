# from openai import OpenAI
# from agent import MakeMyAgent
# from smart_functions import SmartFunction
# from prompt import sentiment_agent_role, outcome_agent_role, assessor_agent_role
# import pandas as pd
# import numpy as np
# import os, json 

# # Initialise client by passing key known to you & Initialise SmartFunction & parameters
# API_KEY = os.getenv(input('Enter your environment variable: '))
# client = OpenAI(api_key=API_KEY) 
# sf = SmartFunction()

# with open("../config/response_format_assessor.json", "r") as f:
#     assessor_response_format = json.load(f)
# with open("../config/response_format_sentiment.json", "r") as f:
#     sentiment_response_format = json.load(f)
# with open("../config/response_format_outcome.json", "r") as f:
#     outcome_response_format = json.load(f)

# # Define model parameters
# model_params = {
#     "temperature": 0.5,
#     "max_tokens": 2048,
#     "top_p": 0.8,
#     "frequency_penalty": 1,
#     "presence_penalty": 1
# }

# # Load data
# df = pd.read_csv('../output/transcripts_labelled.csv')
# df = df.dropna(subset=['ground_positive', 'ground_neutral', 'ground_negative'])
# print(df.head())

# model_sentiments = {}
# limit = len(df)
# for transcript_id, member_body in zip(df['transcript_id'][0:limit], 
#                                       df['member_body'][0:limit]):
#     print(member_body)

#     sentiment_agent = MakeMyAgent(
#         model = "gpt-4o-mini",
#         client = client, 
#         model_params = model_params,
#         role = sentiment_agent_role,
#         response_format = sentiment_response_format,
#         user_input = member_body
#     )
#     predicted_sentiment = sentiment_agent.run()
#     json_predicted_sentiment = json.loads(predicted_sentiment)

#     predictions = {}
#     predictions['positive'] = json_predicted_sentiment['positive']
#     predictions['neutral'] = json_predicted_sentiment['neutral']
#     predictions['negative'] = json_predicted_sentiment['negative']
#     model_sentiments[transcript_id] = predictions


# # Convert dictionary to DataFrame
# predictions_df = pd.DataFrame.from_dict(model_sentiments, orient='index')
# predictions_df.reset_index(inplace=True)
# predictions_df.rename(columns={'index': 'transcript_id', 
#                                'positive': 'predicted_positive', 
#                                'neutral': 'predicted_neutral', 
#                                'negative': 'predicted_negative'}, inplace=True)

# # Get SSE as metric for model
# df = df.merge(predictions_df, on='transcript_id', how='left')
# df['sse'] = df.apply(lambda row: (row['ground_positive'] - row['predicted_positive'])**2 + 
#                                 (row['ground_neutral'] - row['predicted_neutral'])**2 + 
#                                 (row['ground_negative'] - row['predicted_negative'])**2, axis=1)
# df['psuedo_accuracy'] = (1 - df['sse'])/3
# df.to_csv('../output/transcripts_psuedo_optimisation.csv')

# model_sse = np.sum(df['sse'])
# model_pseudo_accuracy = np.mean(df['psuedo_accuracy'])

# print('SSE', model_sse)
# print('Pseudo Accuracy', model_pseudo_accuracy)

import pandas as pd
import numpy as np
import itertools
import json
import os
from openai import OpenAI
from agent import MakeMyAgent
from smart_functions import SmartFunction
from prompt import sentiment_agent_role

class SentimentOptimizer:
    def __init__(self, api_key, data_path, output_path):
        self.client = OpenAI(api_key=api_key)
        self.data_path = data_path
        self.output_path = output_path
        self.df = self.load_data()
        self.model_params_grid = self.create_param_grid()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=['ground_positive', 'ground_neutral', 'ground_negative'])
        return df

    def create_param_grid(self):
        temp = [0.2, 0.8, 1.4]
        top_p = [0.3, 0.6, 0.9]
        fp = [0, 0.5, 1.0]
        pp = [0, 0.5, 1.0]
        return list(itertools.product(temp, top_p, fp, pp))

    def run_sentiment_model(self, member_body, model_params):
        sentiment_agent = MakeMyAgent(
            model="gpt-4o-mini",
            client=self.client,
            model_params=model_params,
            role=sentiment_agent_role,
            response_format=self.load_response_format("../config/response_format_sentiment.json"),
            user_input=member_body
        )
        response = sentiment_agent.run()
        return json.loads(response)

    def load_response_format(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def evaluate_model(self, model_params):
        model_sentiments = {}
        for transcript_id, member_body in zip(self.df['transcript_id'], self.df['member_body']):
            predicted_sentiment = self.run_sentiment_model(member_body, model_params)
            model_sentiments[transcript_id] = {
                'predicted_positive': predicted_sentiment['positive'],
                'predicted_neutral': predicted_sentiment['neutral'],
                'predicted_negative': predicted_sentiment['negative']
            }
        
        predictions_df = pd.DataFrame.from_dict(model_sentiments, orient='index')
        predictions_df.reset_index(inplace=True)
        predictions_df.rename(columns={'index': 'transcript_id'}, inplace=True)
        
        df_merged = self.df.merge(predictions_df, on='transcript_id', how='left')
        df_merged['sse'] = df_merged.apply(lambda row: (row['ground_positive'] - row['predicted_positive'])**2 +
                                                        (row['ground_neutral'] - row['predicted_neutral'])**2 +
                                                        (row['ground_negative'] - row['predicted_negative'])**2, axis=1)
        df_merged['pseudo_accuracy'] = (1 - df_merged['sse']) / 3
        model_sse = np.sum(df_merged['sse'])
        model_pseudo_accuracy = np.mean(df_merged['pseudo_accuracy'])
        
        print(f"Params: {model_params} | SSE: {model_sse:.4f} | Pseudo Accuracy: {model_pseudo_accuracy:.4f}")
        return model_sse, model_pseudo_accuracy

    def run_grid_search(self):
        results = []
        for params in self.model_params_grid:
            model_params = {
                "temperature": params[0],
                "top_p": params[1],
                "frequency_penalty": params[2],
                "presence_penalty": params[3]
            }
            sse, pseudo_accuracy = self.evaluate_model(model_params)
            results.append({"params": model_params, "sse": sse, "pseudo_accuracy": pseudo_accuracy})
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_path, index=False)
        print("Grid search completed. Results saved.")

# Run the optimization
api_key = os.getenv(input('Enter your environment variable: '))
data_path = '../output/transcripts_labelled.csv'
output_path = '../output/grid_search_results.csv'
optimizer = SentimentOptimizer(api_key, data_path, output_path)
optimizer.run_grid_search()
