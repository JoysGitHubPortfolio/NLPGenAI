from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import ast

class SmartPlotter:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()
        self.word_dictionary = self._extract_word_frequencies()
        self._process_data()

    def _extract_word_frequencies(self):
        word_dict = {}
        dominant_emotions = []
        
        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)  # Convert string to dictionary
            positive = dictionary['positive']
            neutral = dictionary['neutral']
            negative = dictionary['negative']
            dominant_emotion = dictionary['dominant_emotion']
            
            dominant_emotions.append(dominant_emotion)
            word_dict[dominant_emotion] = word_dict.get(dominant_emotion, 0) + 1
        
        self.dominant_emotions = dominant_emotions
        return word_dict

    def _process_data(self):
        sentiment_scores = {word: self.sia.polarity_scores(word)['compound'] for word in self.word_dictionary.keys()}
        counts = {word: np.log10(1 + count) for word, count in self.word_dictionary.items()}

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
        colors = ['orange' if s == 0 else 'green' if s > 0 else 'red' for s in self.sentiment_scores_scaled]
        plt.scatter(counts, self.sentiment_scores_scaled, color=colors, edgecolors='black', s=100)

        for i, word in enumerate(words):
            plt.text(counts[i], self.sentiment_scores_scaled[i], word, fontsize=10, ha='right', va='bottom', rotation=30)

        plt.xlabel('Log Emotion Frequency')
        plt.ylabel('Standardized Sentiment Score')
        plt.title('Top Emotions: Frequency vs. Sentiment Scores')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    
    def make_word_map(self):
        wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=self._color_function) \
            .generate_from_frequencies(self.word_dictionary)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Map of Top Emotions (Color + Opacity)')
        plt.show()
    
    def improvement_pie(self, field):
        sentiment_improvement = [1 if x else 0 for x in self.df[field]]
        true_count = sentiment_improvement.count(1)
        false_count = sentiment_improvement.count(0)

        labels = ['Improvement', 'No Improvement']
        sizes = [true_count, false_count]
        colors = ['red', 'green']
        explode = (0.1, 0)  # Slightly explode 'Improvement' for emphasis

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{field} Distribution')
        plt.axis('equal')
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

        plt.hist(positive_sentiments, color='green', label='Positive', alpha=0.5)
        plt.hist(neutral_sentiments, color='orange', label='Neutral', alpha=0.5)
        plt.hist(negative_sentiments, color='red', label='Negative', alpha=0.5)
        plt.legend()
        plt.show()


