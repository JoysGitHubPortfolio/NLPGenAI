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
        elif metric == 'specificity':
            # Specificity = TN / (TN + FP)
            correction_distance = 10
            tns = np.array([confusion_matrix(y_true, (probs >= t).astype(int))[0, 0] for t in thresholds])
            fps = np.array([confusion_matrix(y_true, (probs >= t).astype(int))[0, 1] for t in thresholds])
            
            # Apply epsilon to avoid zero division or extreme values
            specificity = tns / (tns + fps)  # Add epsilon to denominator
            best_threshold = thresholds[np.argmax(specificity)]
            if np.isinf(best_threshold):
                best_threshold = thresholds[np.argsort(specificity)[-correction_distance]]  # second best threshold
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
        # Train models and get probabilities, thresholds, and predictions
        X, y = self.train_models()

        # Find optimal thresholds for each metric for LogReg and NB
        best_thresh_logreg = {
            'recall': self.find_best_threshold(self.logreg_probs, y, metric='recall'),
            'accuracy': self.find_best_threshold(self.logreg_probs, y, metric='accuracy'),
            'specificity': self.find_best_threshold(self.logreg_probs, y, metric='specificity')
        }
        
        best_thresh_nb = {
            'recall': self.find_best_threshold(self.nb_probs, y, metric='recall'),
            'accuracy': self.find_best_threshold(self.nb_probs, y, metric='accuracy'),
            'specificity': self.find_best_threshold(self.nb_probs, y, metric='specificity')
        }

        # Generate predictions
        methods = ['default', 'recall', 'specificity', 'accuracy']
        thresholds = {method: 0.5 if method == 'default' else best_thresh_logreg[method] for method in methods}

        logreg_preds = {method: (self.logreg_probs >= thresholds[method]).astype(int) for method in methods}
        nb_preds = {method: (self.nb_probs >= (0.5 if method == 'default' else best_thresh_nb[method])).astype(int) for method in methods}

        # Compute confusion matrices
        from sklearn.metrics import confusion_matrix
        cm_logreg = {method: confusion_matrix(y, logreg_preds[method]) for method in methods}
        cm_nb = {method: confusion_matrix(y, nb_preds[method]) for method in methods}

        # Helper function to compute metrics
        def compute_metrics(cm):
            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity}

        # Create figure
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        main_title = f"Confusion Matrices & Metrics\n(Blue=LogReg, Green=GNB)\nThresholds Optimised for Each Metric"
        plt.suptitle(main_title, fontsize=14)
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

        for i, method in enumerate(methods):
            # LogReg Confusion Matrix (Blues)
            sns.heatmap(cm_logreg[method], annot=True, cbar=False, fmt='d', cmap='Blues', ax=axes[i, 0], annot_kws={"size": 8})
            axes[i, 0].set_xlabel("Predicted", fontsize=8)
            axes[i, 0].set_ylabel("Actual", fontsize=8)

            # LogReg Metrics Bar Chart
            logreg_metrics = compute_metrics(cm_logreg[method])
            axes[i, 1].bar(logreg_metrics.keys(), logreg_metrics.values(), color='skyblue')
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].tick_params(axis='x', labelrotation=45, labelsize=8)

            # Naive Bayes Confusion Matrix (Greens)
            sns.heatmap(cm_nb[method], annot=True, cbar=False, fmt='d', cmap='Greens', ax=axes[i, 2], annot_kws={"size": 8})
            axes[i, 2].set_xlabel("Predicted", fontsize=8)
            axes[i, 2].set_ylabel("Actual", fontsize=8)

            # Naive Bayes Metrics Bar Chart
            nb_metrics = compute_metrics(cm_nb[method])
            axes[i, 3].bar(nb_metrics.keys(), nb_metrics.values(), color='lightgreen')
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].tick_params(axis='x', labelrotation=45, labelsize=8)

            # Update titles
            logreg_thresh = 0.5 if method == 'default' else {
                'recall': best_thresh_logreg['recall'],
                'specificity': best_thresh_logreg['specificity'],
                'accuracy': best_thresh_logreg['accuracy']
            }[method]

            nb_thresh = 0.5 if method == 'default' else {
                'recall': best_thresh_nb['recall'],
                'specificity': best_thresh_nb['specificity'],
                'accuracy': best_thresh_nb['accuracy']
            }[method]

            # Titles
            logreg_title = f"{method.capitalize()}: threshold = {logreg_thresh:.2f}"
            nb_title = f"{method.capitalize()}: threshold = {nb_thresh:.2f}"

            # Set titles
            axes[i, 0].set_title(logreg_title, fontsize=10)  # LogReg Confusion Matrix
            axes[i, 2].set_title(nb_title, fontsize=10)  # NB Confusion Matrix

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
