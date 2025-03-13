import pandas as pd
import matplotlib.pyplot as plt
from smart_plots import SmartPlotter
from smart_features import SmartFeatures


# Read
df = pd.read_csv('../output/data/overall_model_output.csv')
print(df)

# Get Descriptive Analytics
plotter = SmartPlotter(df)

plotter.improvement_pie('sentiment_improvement')
plotter.improvement_pie('outcome_improvement')
plotter.plot_sentiment_histogram()
plotter.make_scatter_plot()
plotter.make_word_map()
plotter.plot_effective_sentiment_boxplot()
plotter.plot_sentiment_barchart()
plotter.plot_confusion_matrix()

# Train models and plot ROC-AUC comparison for Logistic Regression vs. Naive Bayes
smart_features = SmartFeatures(df)

logreg_auc, nb_auc = smart_features.plot_roc_auc_comparison()
smart_features.plot_confusion_matrices()
print(f"Logistic Regression AUC: {logreg_auc:.3f}")
print(f"Naive Bayes AUC: {nb_auc:.3f}")