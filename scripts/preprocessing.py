import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from scipy import sparse


# Load dataset with specified encoding to handle any decoding issues
train_df = pd.read_csv('../data/train.csv', encoding='ISO-8859-1')


# Basic text preprocessing function
def preprocess_text(text):
   stop_words = set(stopwords.words('english'))
   text = text.lower()  # Convert to lowercase
   text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
   words = text.split()
   words = [word for word in words if word not in stop_words]  # Remove stopwords
   return " ".join(words)


# Apply text preprocessing to the 'text' column
train_df['clean_text'] = train_df['text'].apply(preprocess_text)


# Train-test split
X = train_df['clean_text']
y = train_df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# TF-IDF Vectorization (limiting to top 5000 features)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)


# Save the sparse matrix directly to disk
sparse.save_npz('../data/X_train_tfidf.npz', X_train_tfidf)
sparse.save_npz('../data/X_val_tfidf.npz', X_val_tfidf)


# Save labels as CSV files
y_train.to_csv('../data/y_train.csv', index=False)
y_val.to_csv('../data/y_val.csv', index=False)


print("Preprocessing complete and data saved.")


