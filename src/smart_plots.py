from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
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

    def plot_effective_sentiment_boxplot(self):
        positive_sentiments = []
        neutral_sentiments = []
        negative_sentiments = []

        for i, string in enumerate(self.df['json_sentiment_output']):
            dictionary = ast.literal_eval(string)
            positive_sentiments.append(dictionary['positive'])
            neutral_sentiments.append(dictionary['neutral'])
            negative_sentiments.append(dictionary['negative'])

        # Creating subplots
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Plot effective sentiment
        ax[0].boxplot(self.effective_sentiments)
        ax[0].set_ylabel("Effective Sentiment Score")
        ax[0].set_title("Distribution of Effective Sentiments")
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Plot positive, neutral, negative sentiment probabilities
        ax[1].boxplot([positive_sentiments, neutral_sentiments, negative_sentiments], labels=['Positive', 'Neutral', 'Negative'])
        ax[1].set_ylabel("Sentiment Probability")
        ax[1].set_title("Distribution of Sentiment Probabilities")
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
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
        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=colors)
        plt.title('Sentiment Distribution (RAG Color Scheme)')
        plt.ylabel('Sentiment Count')
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
        plt.title('Confusion Matrix: Outcome vs. Satisfaction')
        plt.show()



class SmartAnalytics():
    def __init__(self, data):
        self.data = data
    
    def extract_features(self, json_data):
        try:
            # Check if 'satisfaction' is a boolean
            satisfaction = json_data['satisfaction']
            if not isinstance(satisfaction, bool):
                raise ValueError(f"'satisfaction' should be a boolean, got {type(satisfaction)}")
            
            # Append the features: confidence, outcome, and satisfaction
            features = []
            features.append(float(json_data['confidence']))  # Confidence value
            features.append(1 if json_data['outcome'] == 'issue resolved' else 0)  # Outcome as binary (resolved=1, not resolved=0)
            features.append(int(satisfaction))  # 'satisfaction' as 1/0
            return features
        except KeyError as e:
            print(f"KeyError: Missing key in the data: {e}")
            raise
        except Exception as e:
            print(f"Error processing record {json_data}: {e}")
            raise
    
    def bayesian_inference(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)
        
        # Initialize and train the Gaussian Naive Bayes model
        model = GaussianNB()        
        model.fit(features, labels)        
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        
        # Get probabilities for each class (0 and 1)
        probs = model.predict_proba(features)  # This returns a 2D array with probabilities for both classes
        
        # Plot the prediction probabilities for class 1 (True satisfaction)
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(probs)), probs[:,1], color='blue', alpha=0.5)  # probs[:, 1] -> Probability for class 1 (True satisfaction)
        plt.title("Prediction Probabilities (Satisfaction Prediction)")
        plt.xlabel("Sample Index")
        plt.ylabel("Probability of Satisfaction (True)")
        plt.show()
        
        # Plot confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Satisfied', 'Satisfied'], yticklabels=['Not Satisfied', 'Satisfied'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        return accuracy, predictions, probs
    
    # Call Bayesian Inference to predict satisfaction
    def analyze_data(self):
        features = []
        labels = []
        for record in self.data:
            feature_vector = self.extract_features(record)
            features.append(feature_vector)            
            labels.append(1 if record['satisfaction'] else 0)
        return self.bayesian_inference(features, labels)
