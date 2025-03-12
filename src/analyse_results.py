import pandas as pd
from smart_plots import SmartPlotter, SmartAnalytics, ast, sns
from smart_features import SmartFeatures

# Read
df = pd.read_csv('../output/overall_model_output.csv')
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

# Bayesian Inference to predict Satisfaction
data = [ast.literal_eval(string) for string in df['json_outcome_output'].values]
analytics = SmartAnalytics(data)
accuracy, predictions, probs = analytics.analyze_data()
print(f"Accuracy: {accuracy}")
print(f"Predictions: {predictions}")

# Logistic Regression to predict Satisfaction
smart_features = SmartFeatures(df)
accuracy = smart_features.plot_sentiment_impact_on_satisfaction()
smart_features.plot_satisfaction_probabilities()
print(f"Accuracy of sentiment impact model: {accuracy}")
