from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import seaborn as sns

class SmartFeatures:
    def __init__(self, df):
        self.df = df
        self._extract_outcome_and_satisfaction()
        self._extract_sentiment_probabilities()

    def _extract_outcome_and_satisfaction(self):
        outcomes = []
        satisfactions = []
        for string in self.df['json_outcome_output']:
            dictionary = ast.literal_eval(string)  # Convert string to dictionary
            outcome = dictionary.get('outcome', '')
            satisfaction = dictionary.get('satisfaction', False)
            outcomes.append(outcome)
            satisfactions.append(satisfaction)
        
        self.satisfactions = satisfactions
        self.outcomes = outcomes

    def _extract_sentiment_probabilities(self):
        positive_probs = []
        neutral_probs = []
        negative_probs = []
        for string in self.df['json_sentiment_output']:
            dictionary = ast.literal_eval(string)
            positive_probs.append(dictionary['positive'])
            neutral_probs.append(dictionary['neutral'])
            negative_probs.append(dictionary['negative'])
        
        self.sentiment_probs = {
            'positive': positive_probs,
            'neutral': neutral_probs,
            'negative': negative_probs
        }

    def plot_sentiment_impact_on_satisfaction(self):
        # Prepare data for training
        X = pd.DataFrame({
            'positive': self.sentiment_probs['positive'],
            'neutral': self.sentiment_probs['neutral'],
            'negative': self.sentiment_probs['negative']
        })
        y = self.satisfactions
        
        # Train a logistic regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Get predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Plot the coefficients (impact of each sentiment term)
        feature_names = ['Positive', 'Neutral', 'Negative']
        coefficients = model.coef_[0]
        
        plt.figure(figsize=(8, 6))
        plt.bar(feature_names, coefficients, color=['green', 'orange', 'red'])
        plt.title('Impact of Sentiment on Satisfaction')
        plt.ylabel('Coefficient Value (Impact on Satisfaction)')
        plt.xlabel('Sentiment Term')
        plt.show()

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Satisfied', 'Satisfied'], yticklabels=['Not Satisfied', 'Satisfied'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return accuracy

    def plot_satisfaction_probabilities(self):
        # Prepare data for prediction
        X = pd.DataFrame({
            'positive': self.sentiment_probs['positive'],
            'neutral': self.sentiment_probs['neutral'],
            'negative': self.sentiment_probs['negative']
        })
        # Train model and predict satisfaction probabilities
        model = LogisticRegression()
        model.fit(X, self.satisfactions)
        prob_satisfaction = model.predict_proba(X)[:, 1]

        # Plot the probabilities of satisfaction based on sentiment probabilities
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(prob_satisfaction)), prob_satisfaction, color='blue', alpha=0.5)
        plt.title("Predicted Probability of Satisfaction based on Sentiment")
        plt.xlabel("Sample Index")
        plt.ylabel("Probability of Satisfaction")
        plt.show()

        # Find the threshold that maximises the accuracy
        thresholds = [np.round(0.1 * i, 2) for i in range(1, 11)]  # Create thresholds from 0.1 to 1.0
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns
        axes = axes.ravel()  # Flatten the axes for easy indexing
        for idx, threshold in enumerate(thresholds):
            thresholded_predictions = (prob_satisfaction >= threshold).astype(int)
            cm = confusion_matrix(self.satisfactions, thresholded_predictions)            
            accuracy = accuracy_score(self.satisfactions, thresholded_predictions)

            # Plot the confusion matrix in the subplot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Satisfied', 'Satisfied'], 
                        yticklabels=['Not Satisfied', 'Satisfied'], ax=axes[idx])            
            axes[idx].set_title(f"Threshold {threshold}, Accuracy: {accuracy:.3f}")
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
        plt.tight_layout()
        plt.show()

        return prob_satisfaction
