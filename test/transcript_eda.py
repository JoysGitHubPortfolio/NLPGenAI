import matplotlib.pyplot as plt
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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

verbose = input('Do you want to see Member body? (1 = Yes, else No): ')
try:
    verbose = int(verbose)
    if verbose == 1:
        verbose= True
except:
    verbose = False

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
        filtered_words = [token for token in tokens if token not in stop_words]

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
    output_path = f"../output/member_{rand_int}_wordcloud.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')  # Remove axes
    plt.title(f'Member {i}')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# Bar chart of the top N words across all members
all_word_counts = Counter(all_words)
top_n_words = all_word_counts.most_common(50)  
words, counts = zip(*top_n_words)
main_output_path = "../output/overall_word_distribution.png"
plt.figure(figsize=(12, 10))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Words Across All Members')
plt.savefig(main_output_path, bbox_inches='tight')
plt.close()