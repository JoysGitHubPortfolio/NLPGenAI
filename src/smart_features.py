from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast

class SmartFeatures:
    def __init__(self, df):
        self.df = df
        self._extract_outcome_and_satisfaction()
        self._extract_sentiment_probabilities()
        self.logreg = None
        self.nb = None
        self.logreg_probs = None
        self.nb_probs = None

    def _extract_outcome_and_satisfaction(self):
        self.satisfactions = []
        self.outcomes = []
        for string in self.df['json_outcome_output']:
            dictionary = ast.literal_eval(string)
            self.outcomes.append(dictionary.get('outcome', ''))
            self.satisfactions.append(dictionary.get('satisfaction', False))
        self.satisfactions = np.array(self.satisfactions).astype(int)

    def _extract_sentiment_probabilities(self):
        self.sentiment_probs = {'positive': [], 'neutral': [], 'negative': []}
        for string in self.df['json_sentiment_output']:
            dictionary = ast.literal_eval(string)
            self.sentiment_probs['positive'].append(dictionary['positive'])
            self.sentiment_probs['neutral'].append(dictionary['neutral'])
            self.sentiment_probs['negative'].append(dictionary['negative'])
        self.sentiment_probs = {k: np.array(v) for k, v in self.sentiment_probs.items()}

    def _prepare_feature_matrix(self):
        return pd.DataFrame(self.sentiment_probs)

    def train_models(self):
        X = self._prepare_feature_matrix()
        y = self.satisfactions
        
        self.logreg = make_pipeline(StandardScaler(), LogisticRegression())
        self.logreg.fit(X, y)
        self.logreg_probs = self.logreg.predict_proba(X)[:, 1]

        self.nb = GaussianNB()
        self.nb.fit(X, y)
        self.nb_probs = self.nb.predict_proba(X)[:, 1]
        return X, y

    def find_best_threshold(self, probs, y_true, metric='recall'):
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        if metric == 'recall':
            best_threshold = thresholds[np.argmax(tpr)]
        elif metric == 'accuracy':
            accuracies = [(probs >= t).astype(int) == y_true for t in thresholds]
            best_threshold = thresholds[np.argmax([np.mean(acc) for acc in accuracies])]
        return best_threshold

    def plot_roc_auc_comparison(self):
        if self.logreg is None or self.nb is None:
            self.train_models()
        X = self._prepare_feature_matrix()
        y = self.satisfactions

        logreg_probs = self.logreg.predict_proba(X)[:, 1]
        nb_probs = self.nb.predict_proba(X)[:, 1]

        fpr_logreg, tpr_logreg, _ = roc_curve(y, logreg_probs)
        fpr_nb, tpr_nb, _ = roc_curve(y, nb_probs)
        logreg_auc = auc(fpr_logreg, tpr_logreg)
        nb_auc = auc(fpr_nb, tpr_nb)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_logreg, tpr_logreg, label=f'LogReg (AUC={logreg_auc:.3f})', color='blue')
        plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={nb_auc:.3f})', color='red')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.show()
        return logreg_auc, nb_auc


    def plot_confusion_matrices(self):
        X, y = self.train_models()

        best_thresh_logreg_recall = self.find_best_threshold(self.logreg_probs, y, metric='recall')
        best_thresh_logreg_acc = self.find_best_threshold(self.logreg_probs, y, metric='accuracy')
        best_thresh_nb_recall = self.find_best_threshold(self.nb_probs, y, metric='recall')
        best_thresh_nb_acc = self.find_best_threshold(self.nb_probs, y, metric='accuracy')

        logreg_preds_recall = (self.logreg_probs >= best_thresh_logreg_recall).astype(int)
        logreg_preds_acc = (self.logreg_probs >= best_thresh_logreg_acc).astype(int)
        nb_preds_recall = (self.nb_probs >= best_thresh_nb_recall).astype(int)
        nb_preds_acc = (self.nb_probs >= best_thresh_nb_acc).astype(int)
        
        logreg_preds_default = (self.logreg_probs >= 0.5).astype(int)
        nb_preds_default = (self.nb_probs >= 0.5).astype(int)

        cm_logreg_recall = confusion_matrix(y, logreg_preds_recall)
        cm_logreg_acc = confusion_matrix(y, logreg_preds_acc)
        cm_nb_recall = confusion_matrix(y, nb_preds_recall)
        cm_nb_acc = confusion_matrix(y, nb_preds_acc)
        cm_logreg_default = confusion_matrix(y, logreg_preds_default)
        cm_nb_default = confusion_matrix(y, nb_preds_default)

        fig, axes = plt.subplots(3, 2, figsize=(6, 8))
        sns.heatmap(cm_logreg_recall, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f"LogReg Max Recall (Threshold: {best_thresh_logreg_recall:.2f})")

        sns.heatmap(cm_nb_recall, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f"Naive Bayes Max Recall (Threshold: {best_thresh_nb_recall:.2f})")

        sns.heatmap(cm_logreg_acc, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f"LogReg Max Accuracy (Threshold: {best_thresh_logreg_acc:.2f})")

        sns.heatmap(cm_nb_acc, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f"Naive Bayes Max Accuracy (Threshold: {best_thresh_nb_acc:.2f})")
        
        sns.heatmap(cm_logreg_default, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
        axes[2, 0].set_title("LogReg Default Threshold (0.5)")
        
        sns.heatmap(cm_nb_default, annot=True, fmt='d', cmap='Blues', ax=axes[2, 1])
        axes[2, 1].set_title("Naive Bayes Default Threshold (0.5)")
        
        for ax in axes.flat:
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        # plt.tight_layout()
        plt.show()
