import pandas as pd  # type: ignore
import os
import numpy as np
import matplotlib  # type: ignore
# Set backend to TkAgg to ensure plots window appears
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from wordcloud import WordCloud, STOPWORDS  # type: ignore
nltk.download('stopwords')
nltk.download('wordnet')

# load the dataset
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "news.csv")
news_d = pd.read_csv(csv_path)

# Normalize labels if they are strings (e.g. 'FAKE'/'REAL') so filtering works later
if news_d['label'].dtype == object:
    # Map common string labels to 1 (Fake) and 0 (Real)
    label_map = {'FAKE': 1, 'Fake': 1, 'REAL': 0, 'Real': 0}
    news_d['label'] = news_d['label'].map(label_map).fillna(0).astype(int)

print("Shape of News data:", news_d.shape)
print("News data columns", news_d.columns)

# by using df.head(), we can immediately familiarize ourselves with the dataset. 
print(news_d.head())

#Text Word startistics: min.mean, max and interquartile range
txt_length = news_d.text.str.split().str.len()
print(txt_length.describe())

#Title statistics 
title_length = news_d.title.str.split().str.len()
print(title_length.describe())

sns.countplot(x="label", data=news_d);
print("1: Unreliable")
print("0: Reliable")
print("Distribution of labels:")
print(news_d.label.value_counts());
print(round(news_d.label.value_counts(normalize=True),2)*100);

# Constants that are used to sanitize the datasets 

column_n = ['id', 'title', 'author', 'text', 'label']
remove_c = ['id','author']
categorical_features = []
target_col = ['label']
text_f = ['title', 'text']

# Clean Datasets
import re
from nltk.stem.porter import PorterStemmer  # type: ignore
from collections import Counter

ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# Removed unused clumns
def remove_unused_c(df,column_n=remove_c):
    # Only drop columns that exist to prevent errors
    df = df.drop([c for c in column_n if c in df.columns], axis=1)
    return df

# Impute null values with None
def null_process(feature_df):
    for col in text_f:
        feature_df.loc[feature_df[col].isnull(), col] = "None"
    return feature_df

def clean_dataset(df):
    # remove unused column
    df = remove_unused_c(df)
    #impute null values
    df = null_process(df)
    return df

# Cleaning text from unused characters
def clean_text(text):
    text = str(text)
    text = re.sub(r'http[\w:/\.]+', ' ', text)  # removing urls
    text = re.sub(r'[^\.\w\s]', ' ', text)  # remove everything but characters and punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    text = text.lower().strip()
    #text = ' '.join(text)    
    return text

## Nltk Preprocessing include:
# Stop words, Stemming and Lemmetization
# For our project we use only Stop word removal
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    #text = ' '.join([word for word in wordlist if word not in stopwords_dict])
    #text = [ps.stem(word) for word in wordlist if not word in stopwords_dict]
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text

# Perform data cleaning on train and test dataset by calling clean_dataset function
df = clean_dataset(news_d)
# apply preprocessing on text through apply method by calling the function nltk_preprocess
df["text"] = df.text.apply(nltk_preprocess)
# apply preprocessing on title through apply method by calling the function nltk_preprocess
df["title"] = df.title.apply(nltk_preprocess)

# Dataset after cleaning and preprocessing step
print(df.head())

# initialize the word cloud
wordcloud = WordCloud( background_color='black', width=800, height=600)

if not df['text'].empty:
    # generate the word cloud by passing the corpus
    text_cloud = wordcloud.generate(' '.join(df['text']))
    # plotting the word cloud
    plt.figure(figsize=(20,30))
    plt.imshow(text_cloud)
    plt.axis('off')

fake_n = ' '.join(df[df['label']==1]['text'])

# Check if there is any fake news text before generating wordcloud
if len(fake_n) > 0:
    wc= wordcloud.generate(fake_n)
    plt.figure(figsize=(20,30))
    plt.imshow(wc)
    plt.axis('off')
    # plt.show()

true_n = ' '.join(df[df['label']==0]['text'])

def plot_top_ngrams(corpus, title, ylabel, xlabel="Number of Occurences", n=2):
  """Utility function to plot top n-grams"""
  if not corpus.strip():
      print(f"No text data available for: {title}")
      return

  # Convert generator to list to ensure pandas Series creation works
  print(f"Generating plot for: {title}")
  true_b = (pd.Series(list(nltk.ngrams(corpus.split(), n))).value_counts())[:20]
  
  # Create a new figure explicitly
  fig, ax = plt.subplots(figsize=(12, 8))
  true_b.sort_values(ascending=False).plot.barh(color='blue', width=.9, ax=ax)
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.set_xlabel(xlabel)
  plt.tight_layout()
  # plt.show()

plot_top_ngrams(true_n, 'Top 20 Frequently Occuring True news Bigrams', "Bigram", n=2)
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Trigrams', "Trigrams", n=3)
plot_top_ngrams(true_n, 'Top 20 Frequently Occuring True news Trigrams', "Trigrams", n=3)
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Trigrams', "Trigrams", n=3)
plt.show()
