import matplotlib.pyplot as plt
import numpy as np
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from collections import Counter
from ssl_verification import *


def check_nltk(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

check_nltk('corpora/stopwords')
check_nltk('tokenizers/punkt')
check_nltk('vader_lexicon')  # Check for VADER lexicon

def make_scatter_plot(log, counts, sentiment_scores, main_output_path):
    scaled = (sentiment_scores - np.mean(sentiment_scores))/np.std(sentiment_scores)
    print(sentiment_scores)
    print(scaled)

    plt.figure(figsize=(10, 10))
    if log:
        counts = [np.log10(1+np.log10(1+x)) for x in counts]
        plt.xlabel('LogLog Word Frequency')
    else:
        plt.xlabel('Word Frequency')
    plt.scatter(counts, scaled,
                color=['orange' if s==0 
                else 'green' if s>0 
                else 'red'
                for s in sentiment_scores],
                edgecolors='black', s=100)

    # Adding words as labels on the scatter plot
    for i, word in enumerate(words):
        plt.text(counts[i], scaled[i], word,
                fontsize=7, ha='left', va='bottom', rotation=90, alpha=0.5)
    plt.ylabel('Standardised Sentiment Score')
    plt.title('Top 50 Words Across All Members: Frequency vs. Sentiment Scores')
    plt.savefig(main_output_path, bbox_inches='tight')
    plt.show()


verbose = input('Do you want to see Member body? (1 = Yes, else No): ')
try:
    verbose = int(verbose)
    if verbose == 1:
        verbose= True
except:
    verbose = False

plot_graph = input('Do you want to see Member word cloud? (1 = Yes, else No): ')
try:
    plot_graph = int(plot_graph)
    if plot_graph == 1:
        plot_graph= True
except:
    plot_graph = False


# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')
custom_stopwords = [
    "hi", "member", "number", "bye",
    "...", ".", ",", "?", "!", "*", "$", "(", ")", 
    "'s", "'m", "'ve", "'d", "id", "'ll", "n't", 
    "mem123456"
]
stop_words += custom_stopwords


# Collect word frequencies for each member
member_dictionary = {}
all_words = []
for i in range(0, 200):
    try:
        with open(f"../transcripts/transcript_{i}.txt") as f:
            text = f.readlines()

        # Filter lines containing "Member"
        member_body = [line.strip().removeprefix('Member: ').lower() 
                    for line in text if 'Member: ' in line]
        member_body = ' '.join(member_body)
        if verbose:
            print(i, member_body)
            print()

        tokens = word_tokenize(member_body)
        filtered_words = [token for token in tokens
                          if token not in stop_words and 'mem' not in token]

        member_body_distribution = {}
        for word in filtered_words:
            if word not in member_body_distribution:
                member_body_distribution[word] = 1
            else:
                member_body_distribution[word] += 1
        member_dictionary[i] = member_body_distribution

        # Add words to the all_words list for global analysis
        all_words.extend(filtered_words)
    except Exception as e:
        print(f"Error at transcript_{i}: {e}")
        continue


# Visualize word clouds for each member
if plot_graph:
    plot_limit = 5
    member_top_n = 10
    for j in range(plot_limit):
        rand_int = random.randint(0, len(member_dictionary.keys()))
        word_distribution = member_dictionary[rand_int]
        member_word_counts = Counter(word_distribution)
        top_n_member_words = member_word_counts.most_common(member_top_n)    
        top_n_word_distribution = dict(top_n_member_words)

        if j == plot_limit:
            break
        wc = WordCloud(width=800,
                    height=400,
                    background_color='white').generate_from_frequencies(top_n_word_distribution)
        output_path = f"../output/plots/member_{rand_int}_wordcloud.png"
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')  # Remove axes
        plt.title(f'Member {rand_int}')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


# Bar chart of the top N words across all members with sentiment analysis
all_word_counts = Counter(all_words)
top_n_words = all_word_counts.most_common(50)  # Adjust N as needed
words, counts = zip(*top_n_words)
sentiment_scores = []
for word in words:
    sentiment_score = sia.polarity_scores(word)['compound']  # 'compound' overall sentiment
    sentiment_scores.append(sentiment_score)

plt.figure(figsize=(8, 8))
bars = plt.barh(words, counts, color=['orange' if s==0 
                                      else 'green' if s>0 
                                      else 'red'
                                      for s in sentiment_scores])
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{sentiment_scores[i]:.2f}', 
             va='center', ha='left', fontsize=10, color='black')
main_output_path = "../output/plots/overall_word_distribution_with_sentiment.png"
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Top 50 Words Across All Members with Sentiment Scores')
plt.savefig(main_output_path, bbox_inches='tight')
plt.close()


# scatter plot and log scale
main_output_path = "../output/plots/overall_word_distribution_scatter_with_sentiment.png"
make_scatter_plot(log=False,
                  counts=counts, 
                  sentiment_scores=sentiment_scores, 
                  main_output_path=main_output_path)

main_output_path_log = "../output/plots/overall_word_distribution_scatter_with_sentiment_log.png"
make_scatter_plot(log=True,
                  counts=counts, 
                  sentiment_scores=sentiment_scores, 
                  main_output_path=main_output_path_log)
