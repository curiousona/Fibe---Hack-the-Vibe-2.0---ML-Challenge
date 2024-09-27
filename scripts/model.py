import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer


# Load your raw training data
try:
   train_df = pd.read_csv('../data/train.csv', encoding='utf-8')
except UnicodeDecodeError:
   train_df = pd.read_csv('../data/train.csv', encoding='ISO-8859-1')


X_train = train_df['text']  # Adjust according to your column name
y_train = train_df['target']  # Adjust according to your column name


# Load validation data from the preprocessed files
X_val = load_npz('../data/X_val_tfidf.npz')
y_val = pd.read_csv('../data/y_val.csv').values.ravel()  # Load validation labels


# Create and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)


# No need to transform X_val since it's already in sparse format
# X_val_tfidf = tfidf.transform(X_val)


# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Validate the model
y_val_pred = model.predict(X_val)
f1 = f1_score(y_val, y_val_pred, average='weighted')


print(f"Validation F1 Score: {f1}")


# Ensure the models directory exists
os.makedirs('../models', exist_ok=True)


# Save the model and the TF-IDF vectorizer
joblib.dump(model, '../models/logistic_model.pkl')
joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')
print("Model and TF-IDF vectorizer saved.")


