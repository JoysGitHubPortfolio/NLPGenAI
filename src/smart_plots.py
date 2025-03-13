from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ast


class SmartPlotter:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()
        self.word_dictionary = self._extract_word_frequencies()
        self._process_data()
        self._extract_outcome_and_satisfaction()
        self.fig_path = '../output/analysis/'

    def _extract_word_frequencies(self):
        word_dict = {}
        dominant_emotions = []
        effective_sentiments = []
        
        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)  # Convert string to dictionary
            positive = dictionary['positive']
            neutral = dictionary['neutral']
            negative = dictionary['negative']
            dominant_emotion = dictionary['dominant_emotion']
            dominant_emotion_intensity = dictionary['dominant_emotion_intensity']

            effective_sentiment = ((positive * 1) + (neutral * 0) + (negative * -1)) * dominant_emotion_intensity
            effective_sentiments.append(effective_sentiment)

            dominant_emotions.append(dominant_emotion)
            word_dict[dominant_emotion] = word_dict.get(dominant_emotion, 0) + 1
        
        self.dominant_emotions = dominant_emotions
        self.effective_sentiments = effective_sentiments  # Store effective sentiments
        return word_dict

    def _process_data(self):
        sentiment_scores = {word: self.sia.polarity_scores(word)['compound'] for word in self.word_dictionary.keys()}
        counts = {word: np.log10(1+count) for word, count in self.word_dictionary.items()}

        self.sentiment_scores = sentiment_scores  # Store raw sentiment scores
        self.counts = counts  # Store log-scaled frequencies

        # Standardize sentiment scores
        scores_array = np.array(list(sentiment_scores.values()))
        self.sentiment_scores_scaled = (scores_array - np.mean(scores_array)) / np.std(scores_array)
    
    def _color_function(self, word, **kwargs):
        score = self.sentiment_scores.get(word, 0)
        magnitude = abs(score)
        lightness = int(70 - (magnitude * 65))  # Scale lightness (90 = faint, 50 = strong)

        if score > 0:
            return f"hsl(120, 100%, {lightness}%)"  # Green
        elif score < 0:
            return f"hsl(0, 100%, {lightness}%)"    # Red
        else:
            return f"hsl(39, 100%, {lightness}%)"   # Orange (neutral)
    
    def make_scatter_plot(self):
        words = list(self.word_dictionary.keys())
        counts = list(self.counts.values())

        plt.figure(figsize=(12, 8))
        colors = ['orange' if s == 0 else 'green' if s > 0 else 'red' for s in self.sentiment_scores.values()]
        plt.scatter(counts, self.sentiment_scores_scaled, color=colors, edgecolors='black', s=100)

        for i, word in enumerate(words):
            plt.text(counts[i], self.sentiment_scores_scaled[i], word, fontsize=10, ha='right', va='bottom', rotation=30)

        fig_title = "Scatter Plot: Emotion Frequency vs Self-associated Sentiment"
        plt.xlabel('Log Emotion Frequency')
        plt.ylabel('Standardized Sentiment Score')
        plt.title(fig_title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()
    
    def make_word_map(self):
        wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=self._color_function) \
            .generate_from_frequencies(self.word_dictionary)

        fig_title = "Top Emotions Mapping\nOpacity Mapped to Emotion Intensity"
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(fig_title)
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()
    

    def improvement_pie(self, field):
        sentiment_improvement = [1 if x else 0 for x in self.df[field]]
        true_count = sentiment_improvement.count(1)
        false_count = sentiment_improvement.count(0)

        labels = ['Required', 'Not Required']
        sizes = [true_count, false_count]
        colors = ['red', 'green']
        explode = (0.1, 0)  # Slightly explode 'Improvement' for emphasis

        fig_title = f"Distribution of {field}"
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(fig_title)
        plt.axis('equal')
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()

    def plot_sentiment_histogram(self):
        positive_sentiments = []
        neutral_sentiments = []
        negative_sentiments = []

        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)
            positive_sentiments.append(dictionary['positive'])
            neutral_sentiments.append(dictionary['neutral'])
            negative_sentiments.append(dictionary['negative'])

        fig_title = 'Sentiment Frequency Distributions across All Classes'
        plt.hist(positive_sentiments, color='green', label='Positive', alpha=0.5)
        plt.hist(neutral_sentiments, color='orange', label='Neutral', alpha=0.5)
        plt.hist(negative_sentiments, color='red', label='Negative', alpha=0.5)
        plt.title(fig_title)
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()

    def plot_effective_sentiment_boxplot(self):
        # Lists for grouped distributions
        maxed_positives = []
        maxed_neutrals = []
        maxed_negatives = []

        # Lists for overall distributions
        positive_sentiments = []
        neutral_sentiments = []
        negative_sentiments = []

        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)
            positive = dictionary['positive']
            neutral = dictionary['neutral']
            negative = dictionary['negative']

            positive_sentiments.append(positive)
            neutral_sentiments.append(neutral)
            negative_sentiments.append(negative)

            sentiments = [positive, neutral, negative]
            if max(sentiments) == positive:
                maxed_positives.append(positive)
            elif max(sentiments) == neutral:
                maxed_neutrals.append(neutral)
            else:
                maxed_negatives.append(negative)

        # Creating subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))

        # Plot effective sentiment
        ax[0].boxplot(self.effective_sentiments)
        ax[0].set_ylabel("Effective Sentiment Score")
        ax[0].set_title("Distribution of Effective Sentiments")
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Plot positive, neutral, negative sentiment probabilities
        ax[1].boxplot([positive_sentiments, neutral_sentiments, negative_sentiments], labels=['Positive', 'Neutral', 'Negative'])
        ax[1].set_ylabel("Sentiment Probability")
        ax[1].set_title("Distribution of Sentiment Probabilities")
        ax[1].set_ylim(-0.05,1)
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Plot positive, neutral, negative sentiment probabilities maxes
        ax[2].boxplot([maxed_positives, maxed_neutrals, maxed_negatives], labels=['Positive', 'Neutral', 'Negative'])
        ax[2].set_ylabel("Sentiment Probability")
        ax[2].set_title("Distribution of Sentiment Probabilities for Maxed Class")
        ax[2].set_ylim(-0.05,1)
        ax[2].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'{self.fig_path}Effective Sentiment Score compared to Maximised Classes.png')
        plt.show()

    def plot_sentiment_barchart(self):
        # Initialize counts for positive, neutral, and negative sentiments
        positive_count = 0
        neutral_count = 0
        negative_count = 0

        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)
            positive_count += dictionary['positive']
            neutral_count += dictionary['neutral']
            negative_count += dictionary['negative']

        # Define counts and labels
        counts = [positive_count, neutral_count, negative_count]
        labels = ['Positive', 'Neutral', 'Negative']

        # Assign color based on RAG scheme
        colors = ['green' if count == positive_count else 'orange' if count == neutral_count else 'red' for count in counts]

        # Create bar chart
        fig_title = 'Sentiment Distribution'
        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=colors)
        plt.title(fig_title)
        plt.ylabel('Count')
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()

    def _extract_outcome_and_satisfaction(self):
            outcomes = []
            satisfactions = []
            
            for string in self.df['json_outcome_output']:
                dictionary = ast.literal_eval(string)  # Convert string to dictionary
                outcome = dictionary.get('outcome', '')
                satisfaction = dictionary.get('satisfaction', False)
                
                outcomes.append(outcome)
                satisfactions.append(satisfaction)
            
            return outcomes, satisfactions
        
    def plot_confusion_matrix(self):
        # Extract 'outcome' and 'satisfaction' data
        outcomes, satisfactions = self._extract_outcome_and_satisfaction()
        df_matrix = pd.DataFrame({
            'Outcome': outcomes,
            'Satisfaction': satisfactions
        })
        confusion_matrix = pd.crosstab(df_matrix['Outcome'], df_matrix['Satisfaction'], 
                                    rownames=['Outcome'], colnames=['Satisfaction'])

        # Plot the confusion matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, 
                    linewidths=0.5, linecolor='black')
        fig_title = 'Confusion Matrix: Outcome vs. Satisfaction'
        plt.title(fig_title)
        plt.savefig(f"{self.fig_path}{fig_title}.png")
        plt.show()