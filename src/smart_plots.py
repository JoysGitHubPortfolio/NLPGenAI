import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

class SmartPlotter:
    def __init__(self, word_dictionary):
        self.word_dictionary = word_dictionary
        self.sia = SentimentIntensityAnalyzer()
        self.words = list(word_dictionary.keys())
        self.sentiment_scores = []
        self.counts = []
        self._process_data()

    def _process_data(self):
        for word, count in self.word_dictionary.items():
            sentiment_score = self.sia.polarity_scores(word)['compound']
            self.sentiment_scores.append(sentiment_score)
            self.counts.append(count)

        # Convert to numpy arrays
        self.sentiment_scores = np.array(self.sentiment_scores)
        self.counts = np.log10(1 + np.array(self.counts))  # Log scale to handle frequency
        self.sentiment_scores_scaled = (self.sentiment_scores - np.mean(self.sentiment_scores)) / np.std(self.sentiment_scores)

    def make_scatter_plot(self):
        colors = ['orange' if s == 0 else 'green' if s > 0 else 'red' for s in self.sentiment_scores]
        plt.figure(figsize=(12, 8))
        plt.scatter(self.counts, self.sentiment_scores_scaled, color=colors, edgecolors='black', s=100)
        
        # Add word labels
        for i, word in enumerate(self.words):
            plt.text(self.counts[i], self.sentiment_scores_scaled[i], word, fontsize=10, ha='right', va='bottom', rotation=30)
        
        plt.xlabel('Log Emotion Frequency')
        plt.ylabel('Standardised Sentiment Score')
        plt.title('Top Emotions: Frequency vs. Sentiment Scores')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    
    def make_word_map(self):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.word_dictionary)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Map of Top Emotions')
        plt.show()

