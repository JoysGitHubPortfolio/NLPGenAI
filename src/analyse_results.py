import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read
df = pd.read_csv('../output/overall_model_output.csv')
print(df)

# Create list
def improvement_pie(df, field):
    sentiment_improvement = [1 if x == True else 0 for x in df[field]]
    true_count = sentiment_improvement.count(1)
    false_count = sentiment_improvement.count(0)

    # Pie chart
    labels = ['Improvement', 'No Improvement']
    sizes = [true_count, false_count]
    colors = ['red', 'green'] 
    explode = (0.1, 0)  # Slightly explode 'Improvement' for emphasis

    # Plotting the pie chart
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'{field} Distribution')
    plt.axis('equal')
    plt.show()

# improvement_pie(df=df, field='sentiment_improvement')
# improvement_pie(df=df, field='outcome_improvement')

positive_sentiments = []
neutral_sentiments = []
negative_sentiments = []
dominant_emotions = []
for i, string in enumerate(df['json_sentiment_output']):
    dictionary = ast.literal_eval(string) # fixes annoyance about speech marks!
    print(i, dictionary['positive'])
    positive_sentiments.append(dictionary['positive'])
    neutral_sentiments.append(dictionary['neutral'])
    negative_sentiments.append(dictionary['negative'])
    dominant_emotions.append(dictionary['dominant_emotion'])

plt.hist(positive_sentiments, color='green', label='Positive', alpha=0.5)
plt.hist(neutral_sentiments, color='orange', label='Neutral', alpha=0.5)
plt.hist(negative_sentiments, color='red', label='Negative', alpha=0.5)
plt.legend()
plt.show()

emotions_count = {}
for e in dominant_emotions:
    if e not in emotions_count:
        emotions_count[e] = 1
    else:
        emotions_count[e] += 1  # Corrected line

# Manual duplication removal
gratitude_count = emotions_count.get('grateful', 0) + emotions_count.get('gratefulness', 0) + emotions_count.get('gratitude', 0)
emotions_count.pop('grateful', None)
emotions_count.pop('gratefulness', None)
emotions_count.pop('gratitude', None)
emotions_count['gratitude'] = gratitude_count

# Sort desc
emotions_count = dict(sorted(emotions_count.items(), key=lambda item: item[1], reverse=True))
print(emotions_count)

# Plotting the bar chart
plt.barh(list(emotions_count.keys()), list(emotions_count.values()))
plt.xticks(rotation=90)
plt.show()
