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
        best_thresh_logreg_recall      = self.find_best_threshold(self.logreg_probs, y, metric='recall')
        best_thresh_logreg_acc         = self.find_best_threshold(self.logreg_probs, y, metric='accuracy')
        best_thresh_logreg_specificity = self.find_best_threshold(self.logreg_probs, y, metric='specificity')

        best_thresh_nb_recall      = self.find_best_threshold(self.nb_probs, y, metric='recall')
        best_thresh_nb_acc         = self.find_best_threshold(self.nb_probs, y, metric='accuracy')
        best_thresh_nb_specificity = self.find_best_threshold(self.nb_probs, y, metric='specificity')

        # Generate predictions for each threshold option (default always uses 0.5)
        logreg_preds_default      = (self.logreg_probs >= 0.5).astype(int)
        logreg_preds_recall       = (self.logreg_probs >= best_thresh_logreg_recall).astype(int)
        logreg_preds_specificity  = (self.logreg_probs >= best_thresh_logreg_specificity).astype(int)
        logreg_preds_acc          = (self.logreg_probs >= best_thresh_logreg_acc).astype(int)

        nb_preds_default      = (self.nb_probs >= 0.5).astype(int)
        nb_preds_recall       = (self.nb_probs >= best_thresh_nb_recall).astype(int)
        nb_preds_specificity  = (self.nb_probs >= best_thresh_nb_specificity).astype(int)
        nb_preds_acc          = (self.nb_probs >= best_thresh_nb_acc).astype(int)

        # Compute confusion matrices
        from sklearn.metrics import confusion_matrix
        cm_logreg_default      = confusion_matrix(y, logreg_preds_default)
        cm_logreg_recall       = confusion_matrix(y, logreg_preds_recall)
        cm_logreg_specificity  = confusion_matrix(y, logreg_preds_specificity)
        cm_logreg_acc          = confusion_matrix(y, logreg_preds_acc)

        cm_nb_default      = confusion_matrix(y, nb_preds_default)
        cm_nb_recall       = confusion_matrix(y, nb_preds_recall)
        cm_nb_specificity  = confusion_matrix(y, nb_preds_specificity)
        cm_nb_acc          = confusion_matrix(y, nb_preds_acc)

        # Organize confusion matrices by threshold type
        methods = ['default', 'recall', 'specificity', 'accuracy']
        cm_logreg = {
            'default':      cm_logreg_default,
            'recall':       cm_logreg_recall,
            'specificity':  cm_logreg_specificity,
            'accuracy':     cm_logreg_acc
        }
        cm_nb = {
            'default':      cm_nb_default,
            'recall':       cm_nb_recall,
            'specificity':  cm_nb_specificity,
            'accuracy':     cm_nb_acc
        }

        # Helper function to compute metrics from a 2x2 confusion matrix
        def compute_metrics(cm):
            # Assuming cm is [[TN, FP], [FN, TP]]
            TN, FP, FN, TP = cm.ravel()
            accuracy    = (TP + TN) / (TP + TN + FP + FN)
            precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            return {'Accuracy': accuracy, 
                    'Precision': precision, 
                    'Recall': recall, 
                    'Specificity': specificity}

        # Create a figure with 4 rows and 4 columns:
        # Columns 0 and 1 will be for LogReg (confusion matrix and bar chart)
        # Columns 2 and 3 will be for Naive Bayes
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        plt.subplots_adjust(wspace=0.4, hspace=0.6)

        for i, method in enumerate(methods):
            # ----------------------
            # LogReg Plots (left two columns)
            # ----------------------
            # Confusion matrix (column 0)
            sns.heatmap(cm_logreg[method], annot=True, cbar=False, fmt='d',
                        cmap='Blues', ax=axes[i, 0], annot_kws={"size": 8})
            if method == 'default':
                thresh = 0.5
            else:
                thresh = {
                    'recall':       best_thresh_logreg_recall,
                    'specificity':  best_thresh_logreg_specificity,
                    'accuracy':     best_thresh_logreg_acc
                }[method]
            axes[i, 0].set_title(f"LogReg {method.capitalize()} (Thresh: {thresh:.2f})", fontsize=10)
            axes[i, 0].set_xlabel("Predicted", fontsize=8)
            axes[i, 0].set_ylabel("Actual", fontsize=8)

            # Metrics bar chart (column 1)
            logreg_metrics = compute_metrics(cm_logreg[method])
            axes[i, 1].bar(logreg_metrics.keys(), logreg_metrics.values(), color='skyblue')
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].set_title("LogReg Metrics", fontsize=10)
            axes[i, 1].tick_params(axis='x', labelrotation=45, labelsize=8)

            # ----------------------
            # Naive Bayes Plots (right two columns)
            # ----------------------
            # Confusion matrix (column 2)
            sns.heatmap(cm_nb[method], annot=True, cbar=False, fmt='d',
                        cmap='Blues', ax=axes[i, 2], annot_kws={"size": 8})
            if method == 'default':
                thresh = 0.5
            else:
                thresh = {
                    'recall':       best_thresh_nb_recall,
                    'specificity':  best_thresh_nb_specificity,
                    'accuracy':     best_thresh_nb_acc
                }[method]
            axes[i, 2].set_title(f"NB {method.capitalize()} (Thresh: {thresh:.2f})", fontsize=10)
            axes[i, 2].set_xlabel("Predicted", fontsize=8)
            axes[i, 2].set_ylabel("Actual", fontsize=8)

            # Metrics bar chart (column 3)
            nb_metrics = compute_metrics(cm_nb[method])
            axes[i, 3].bar(nb_metrics.keys(), nb_metrics.values(), color='lightgreen')
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].set_title("NB Metrics", fontsize=10)
            axes[i, 3].tick_params(axis='x', labelrotation=45, labelsize=8)
        plt.tight_layout()
        plt.show()
