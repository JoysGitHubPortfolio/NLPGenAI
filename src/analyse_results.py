import pandas as pd
import matplotlib.pyplot as plt
from smart_plots import SmartPlotter

# Read
df = pd.read_csv('../output/overall_model_output.csv')
print(df)

plotter = SmartPlotter(df)
plotter.improvement_pie('sentiment_improvement')
plotter.improvement_pie('outcome_improvement')
plotter.plot_sentiment_histogram()
plotter.make_scatter_plot()
plotter.make_word_map()