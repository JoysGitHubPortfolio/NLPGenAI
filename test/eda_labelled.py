import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../output/transcripts_labelled.csv')
print(df.info())

df = df.dropna(subset=['ground_positive', 'ground_neutral', 'ground_negative'])
print(df)

plt.hist(df['ground_positive'], color='green', alpha=0.6, label='Positive')
plt.hist(df['ground_neutral'], color='orange', alpha=0.6, label='Neutral')
plt.hist(df['ground_negative'], color='red', alpha=0.6, label='Negative')
plt.xlabel('Probability of Class')
plt.ylabel('Frequency of probalities')
plt.title('Probability Distributions for Ground Truth Labels')
plt.legend()
plt.show()
