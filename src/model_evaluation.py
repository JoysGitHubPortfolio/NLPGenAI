import os, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from agent import MakeMyAgent
from prompt import sentiment_agent_role
from sklearn.metrics import confusion_matrix, classification_report

# Define a function to determine the majority sentiment
def get_majority_sentiment(row):
    sentiment_scores = {
        "positive": row["ground_positive"],
        "neutral": row["ground_neutral"],
        "negative": row["ground_negative"]
    }
    return max(sentiment_scores, key=sentiment_scores.get)

# Load API Key
API_KEY = os.getenv(input('Enter your environment variable: '))
client = OpenAI(api_key=API_KEY) 
api = input('Call GPT? 1 = Yes, Else No: ')
try:
    if int(api) == 1:
        api = True
    else:
        api = False
except:
    api = False

# Load Data
df = pd.read_csv('../output/transcripts_labelled.csv').dropna(subset=['ground_positive'])
df["ground_majority"] = df.apply(get_majority_sentiment, axis=1)
print(df[["ground_positive", "ground_neutral", "ground_negative", "ground_majority"]].head())

# Load Response Formats
with open("../config/response_format_sentiment.json", "r") as f:
    sentiment_response_format = json.load(f)

# Choose the best parameters from grid search (lowest SSE), default if not found
model_params_data = pd.read_csv('../output/grid_search_results.csv')
best_model_params = model_params_data.loc[model_params_data['sse'].idxmin()].to_dict()
best_model_params = {
    "temperature": best_model_params.get("temperature", 0.5),
    "max_tokens": best_model_params.get("max_tokens", 2048),
    "top_p": best_model_params.get("top_p", 0.8),
    "frequency_penalty": best_model_params.get("frequency_penalty", 1),
    "presence_penalty": best_model_params.get("presence_penalty", 1),
}
print("Best Model Parameters:", best_model_params)
print()

# Store predictions
model_predictions = {}
if api:
    for i, input_body in zip(df.index, df["member_body"]):
        sentiment_agent = MakeMyAgent(
            model="gpt-4o-mini",
            client=client, 
            model_params=best_model_params,
            role=sentiment_agent_role,
            response_format=sentiment_response_format,
            user_input=input_body
        )    
        # Handle API Response
        try:
            sentiment_output = sentiment_agent.run()
            json_sentiment_output = json.loads(sentiment_output)
            
            model_predictions[i] = {
                'pred_positive': json_sentiment_output.get('positive', 0),
                'pred_neutral': json_sentiment_output.get('neutral', 0),
                'pred_negative': json_sentiment_output.get('negative', 0),
            }
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            model_predictions[i] = {'pred_positive': None, 'pred_neutral': None, 'pred_negative': None}
predictions_df = pd.DataFrame.from_dict(model_predictions, orient='index')
df = df.join(predictions_df)

# Determine `pred_majority`
df["pred_majority"] = df.apply(get_majority_sentiment, axis=1)
labels = ["positive", "neutral", "negative"]
conf_matrix = confusion_matrix(df["ground_majority"], df["pred_majority"], labels=labels)
class_report = classification_report(df["ground_majority"], df["pred_majority"], target_names=labels, output_dict=True)

# Convert classification report to DataFrame
metrics_df = pd.DataFrame(class_report).T.iloc[:-1, :-1]  # Drop "accuracy" row

# Create a 2-row, 3-column figure layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})

# --- Confusion Matrix Heatmap (Top Left) ---
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("True Label")

# --- Classification Report Bar Chart (Top Right - spans two columns) ---
metrics_df.plot(kind="bar", ax=axes[0, 1], colormap="viridis", edgecolor="black")
axes[0, 1].set_title("Precision, Recall, F1-score")
axes[0, 1].set_xticks(range(len(labels)))  # Set ticks for each label
axes[0, 1].set_xticklabels(labels, rotation=0)
axes[0, 1].legend(["Precision", "Recall", "F1-score"])

# --- Boxplots (Second Row) ---
sentiments = ["positive", "neutral", "negative"]
color_mapping = {
    "ground_positive": "lightcoral", "pred_positive": "royalblue",
    "ground_neutral": "lightcoral", "pred_neutral": "royalblue",
    "ground_negative": "lightcoral", "pred_negative": "royalblue",
}

sse = 0
mae = 0
scaler = StandardScaler()

for i, sentiment in enumerate(sentiments):
    # Compute mean values for ground truth and predictions
    ground_mean = df[f"ground_{sentiment}"].mean()
    pred_mean = df[f"pred_{sentiment}"].mean()
    
    sse += (ground_mean - pred_mean) ** 2
    mae += abs(ground_mean - pred_mean)
    print(f"{sentiment.capitalize()} Mean Scores -> Ground Truth: {ground_mean:.4f}, Predicted: {pred_mean:.4f}")

    # Prepare data for boxplot
    subset = df.melt(
        value_vars=[f"ground_{sentiment}", f"pred_{sentiment}"],
        var_name="Type",
        value_name="Score"
    )
    # Standardize the data (scale it)
    subset["Score"] = scaler.fit_transform(subset[["Score"]])

    # Create boxplot with corrected hue mapping
    sns.boxplot(data=subset, x="Type", y="Score", hue="Type", palette=color_mapping, ax=axes[1, i], dodge=False)
    axes[1, i].set_title(f"{sentiment.capitalize()} Sentiment")
    axes[1, i].set_xlabel("")
    axes[1, i].set_xticklabels(["Ground Truth", "Predicted"])
    axes[1, i].set_ylabel("Standardized Sentiment Score" if i == 0 else "")

# --- Bar chart for SSE and MAE ---
metrics = {'SSE': sse, 'MAE': mae}
axes[0, 2].bar(metrics.keys(), metrics.values(), color=['lightblue', 'salmon'])
axes[0, 2].set_title("SSE & MAE Comparison")
axes[0, 2].set_ylabel("Error Value")
axes[0, 2].set_xlabel("Metric")

# Adjust layout and show plot
plt.suptitle("Model Evaluation: Confusion Matrix, Classification Report & Sentiment Distributions", fontsize=14)
plt.tight_layout()
plt.savefig('../output/model_evaluation.png', dpi=300)
plt.show()

# Print error metrics
print(f"\nSum of Squared Errors (SSE) between mean sentiment scores: {sse:.4f}")
print(f"Mean Absolute Error (MAE) between mean sentiment scores: {mae:.4f}")
