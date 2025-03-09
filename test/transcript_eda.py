import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')
custom_stopwords = [
    "hi", "member", "number", "bye", "...", ".", ",", "?", "!", "*", "$", "(", ")", "'s", "'m", "'ve", "'d", "id", "'ll", "n't", "mem123456"
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
limit = 5
for i, word_distribution in member_dictionary.items():
    if i == limit:
        break
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_distribution)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Member {i} - Word Cloud')
    plt.show()

all_word_counts = Counter(all_words)
top_n_words = all_word_counts.most_common(50)  # Change to any number N you want
words, counts = zip(*top_n_words)

# Bar chart of the top N words across all members
plt.figure(figsize=(12, 6))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Words Across All Members')
plt.show()
